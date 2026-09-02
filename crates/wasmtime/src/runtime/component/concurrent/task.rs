use super::{AsAccessor, GuestTaskId, TaskId};
use crate::store::StoreId;
use crate::try_mutex::TryMutex;
use crate::{AsContextMut, Result};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

/// A host-owned handle to a guest task created by `start_call_concurrent`.
///
/// This handle is available immediately after calling
/// [`Func::start_call_concurrent`](crate::component::Func::start_call_concurrent)
/// or
/// [`TypedFunc::start_call_concurrent`](crate::component::TypedFunc::start_call_concurrent).
/// It may be cloned, used to request cancellation with
/// [`GuestTaskHandle::cancel`], and used to wait for the task to produce a
/// terminal result and for its implicit thread to exit with
/// [`GuestTaskHandle::task_done`].
///
/// Dropping this handle does not affect the guest task.
#[derive(Clone)]
pub struct GuestTaskHandle {
    store: StoreId,
    task: TaskId,
    // Like `JoinHandle`, this lock is only accessed while the owning store's
    // event loop serially polls work. Contention therefore indicates a runtime
    // bug.
    state: GuestTaskHandleState,
}

/// Error returned by a concurrent call's result future when the host cancels
/// the task before parameter lowering begins or when the guest acknowledges a
/// cancellation request by calling the `task.cancel` intrinsic.
///
/// After parameter lowering begins, a guest may instead ignore the request or
/// call `task.return`, in which case the call result is returned normally.
#[derive(Debug)]
pub struct GuestTaskCancelled;

impl fmt::Display for GuestTaskCancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("guest task was cancelled")
    }
}

impl core::error::Error for GuestTaskCancelled {}

#[derive(Clone)]
pub(super) struct GuestTaskHandleState {
    state: Arc<TryMutex<GuestTaskState>>,
}

enum GuestTaskState {
    Running {
        waiters: Vec<Option<Waker>>,
        free_waiters: Vec<usize>,
    },
    Complete,
}

/// Future used by [`GuestTaskHandle::task_done`].
///
/// Each pending future owns one waiter slot. Dropping the future unregisters
/// that slot so a task which ignores cancellation cannot retain stale wakers.
struct TaskDone {
    store: StoreId,
    state: GuestTaskHandleState,
    waiter: Option<usize>,
}

impl GuestTaskHandle {
    pub(super) fn new(
        store: StoreId,
        task: TaskId,
        state: GuestTaskHandleState,
    ) -> GuestTaskHandle {
        GuestTaskHandle { store, task, state }
    }

    pub(super) fn task_id(&self) -> TaskId {
        self.task
    }

    /// Returns the diagnostic identifier for the guest task represented by this
    /// handle.
    ///
    /// The returned ID may be correlated with
    /// [`StoreContextMut::async_call_stack`](crate::StoreContextMut::async_call_stack).
    pub fn id(&self) -> GuestTaskId {
        self.task.guest_task_id()
    }

    /// Requests cancellation of this guest task.
    ///
    /// If parameter lowering has not begun, the task is cancelled immediately
    /// and guest code is not entered. Once parameter lowering begins,
    /// cancellation is asynchronous and cooperative: the guest may take an
    /// arbitrary amount of time to observe the request, may ignore it, or may
    /// call `task.return` instead of `task.cancel`.
    ///
    /// This method may be called before or after the call result is produced.
    /// Calling it after [`GuestTaskHandle::task_done`] would complete is a
    /// no-op. Use [`GuestTaskHandle::task_done`] to wait until the task's
    /// implicit thread has exited.
    ///
    /// # Panics
    ///
    /// Panics if `accessor` belongs to a different store than this handle or is
    /// used outside the context in which that accessor is valid. See
    /// [`Accessor::with`](crate::component::Accessor::with).
    pub fn cancel(&self, accessor: impl AsAccessor) -> Result<()> {
        accessor.as_accessor().with(|mut access| {
            let store = access.as_context_mut();
            self.store.assert_belongs_to(store.0.id());

            if self.state.is_complete() {
                Ok(())
            } else {
                self.task.cancel(store.0)
            }
        })
    }

    /// Waits until this guest task has produced a terminal result and its
    /// implicit thread has exited.
    ///
    /// The implicit thread may continue running after it calls `task.return` or
    /// `task.cancel`, so the corresponding call-result future may resolve
    /// before this method does. Explicit threads created by the guest are not
    /// part of this completion condition and may still be running when this
    /// method returns.
    ///
    /// This future must be polled by the owning store's component event loop,
    /// for example within
    /// [`StoreContextMut::run_concurrent`](crate::StoreContextMut::run_concurrent)
    /// or from a concurrent host function registered with
    /// [`LinkerInstance::func_wrap_concurrent`](crate::component::LinkerInstance::func_wrap_concurrent).
    ///
    /// # Panics
    ///
    /// Panics if `accessor` belongs to a different store than this handle or if
    /// this future is polled outside the owning store's component event loop.
    pub async fn task_done(&self, accessor: impl AsAccessor) {
        accessor.as_accessor().with(|mut access| {
            let store = access.as_context_mut();
            self.store.assert_belongs_to(store.0.id());
        });
        drop(accessor);

        TaskDone::new(self.store, self.state.clone()).await
    }
}

impl TaskDone {
    fn new(store: StoreId, state: GuestTaskHandleState) -> TaskDone {
        TaskDone {
            store,
            state,
            waiter: None,
        }
    }
}

impl Future for TaskDone {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        super::check_ambient_store(this.store);
        this.state.poll_complete(&mut this.waiter, cx)
    }
}

impl Drop for TaskDone {
    fn drop(&mut self) {
        if let Some(waiter) = self.waiter.take() {
            self.state.remove_waiter(waiter);
        }
    }
}

impl GuestTaskHandleState {
    pub(super) fn new() -> GuestTaskHandleState {
        GuestTaskHandleState {
            state: Arc::new(TryMutex::new(GuestTaskState::Running {
                waiters: Vec::new(),
                free_waiters: Vec::new(),
            })),
        }
    }

    pub(super) fn is_complete(&self) -> bool {
        matches!(
            &*self.state.try_lock().expect("should not be contended"),
            GuestTaskState::Complete
        )
    }

    fn poll_complete(&self, waiter: &mut Option<usize>, cx: &mut Context<'_>) -> Poll<()> {
        let mut state = self.state.try_lock().expect("should not be contended");
        match &mut *state {
            GuestTaskState::Running {
                waiters,
                free_waiters,
            } => {
                match *waiter {
                    Some(index) => {
                        let registered = waiters
                            .get_mut(index)
                            .and_then(Option::as_mut)
                            .expect("registered waiter should be active");
                        if !registered.will_wake(cx.waker()) {
                            *registered = cx.waker().clone();
                        }
                    }
                    None => {
                        let index = match free_waiters.pop() {
                            Some(index) => {
                                debug_assert!(waiters[index].is_none());
                                waiters[index] = Some(cx.waker().clone());
                                index
                            }
                            None => {
                                waiters.push(Some(cx.waker().clone()));
                                waiters.len() - 1
                            }
                        };
                        *waiter = Some(index);
                    }
                }
                Poll::Pending
            }
            GuestTaskState::Complete => {
                *waiter = None;
                Poll::Ready(())
            }
        }
    }

    fn remove_waiter(&self, waiter: usize) {
        let mut state = self.state.try_lock().expect("should not be contended");
        let GuestTaskState::Running {
            waiters,
            free_waiters,
        } = &mut *state
        else {
            return;
        };

        let slot = waiters
            .get_mut(waiter)
            .expect("registered waiter should exist");
        if slot.take().is_some() {
            free_waiters.push(waiter);
        }
    }

    pub(super) fn complete(&self) {
        let waiters = {
            let mut state = self.state.try_lock().expect("should not be contended");
            match mem::replace(&mut *state, GuestTaskState::Complete) {
                GuestTaskState::Running { waiters, .. } => waiters,
                GuestTaskState::Complete => return,
            }
        };

        for waker in waiters.into_iter().flatten() {
            waker.wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    #[derive(Default)]
    struct WakeCounter {
        wakes: AtomicUsize,
    }

    impl Wake for WakeCounter {
        fn wake(self: Arc<Self>) {
            self.wakes.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn waker(counter: &Arc<WakeCounter>) -> Waker {
        Waker::from(counter.clone())
    }

    fn poll(done: &mut TaskDone, waker: &Waker) -> Poll<()> {
        let mut cx = Context::from_waker(waker);
        done.state.poll_complete(&mut done.waiter, &mut cx)
    }

    fn active_waiters(state: &GuestTaskHandleState) -> usize {
        let state = state.state.try_lock().expect("should not be contended");
        match &*state {
            GuestTaskState::Running { waiters, .. } => {
                waiters.iter().filter(|waiter| waiter.is_some()).count()
            }
            GuestTaskState::Complete => 0,
        }
    }

    fn waiter_slots(state: &GuestTaskHandleState) -> usize {
        let state = state.state.try_lock().expect("should not be contended");
        match &*state {
            GuestTaskState::Running { waiters, .. } => waiters.len(),
            GuestTaskState::Complete => 0,
        }
    }

    #[test]
    fn task_done_waiter_is_removed_when_dropped() {
        let state = GuestTaskHandleState::new();
        let first_counter = Arc::new(WakeCounter::default());
        let first_waker = waker(&first_counter);
        let mut first = TaskDone::new(StoreId::allocate(), state.clone());

        assert_eq!(poll(&mut first, &first_waker), Poll::Pending);
        assert_eq!(active_waiters(&state), 1);
        assert_eq!(waiter_slots(&state), 1);

        drop(first);
        assert_eq!(active_waiters(&state), 0);

        let second_counter = Arc::new(WakeCounter::default());
        let second_waker = waker(&second_counter);
        let mut second = TaskDone::new(StoreId::allocate(), state.clone());
        assert_eq!(poll(&mut second, &second_waker), Poll::Pending);
        assert_eq!(active_waiters(&state), 1);
        assert_eq!(waiter_slots(&state), 1);

        state.complete();
        assert_eq!(first_counter.wakes.load(Ordering::SeqCst), 0);
        assert_eq!(second_counter.wakes.load(Ordering::SeqCst), 1);
        assert_eq!(poll(&mut second, &second_waker), Poll::Ready(()));
    }

    #[test]
    fn task_done_updates_its_registered_waker() {
        let state = GuestTaskHandleState::new();
        let first_counter = Arc::new(WakeCounter::default());
        let second_counter = Arc::new(WakeCounter::default());
        let first_waker = waker(&first_counter);
        let second_waker = waker(&second_counter);
        let mut done = TaskDone::new(StoreId::allocate(), state.clone());

        assert_eq!(poll(&mut done, &first_waker), Poll::Pending);
        assert_eq!(poll(&mut done, &second_waker), Poll::Pending);
        assert_eq!(active_waiters(&state), 1);

        state.complete();
        state.complete();

        assert_eq!(first_counter.wakes.load(Ordering::SeqCst), 0);
        assert_eq!(second_counter.wakes.load(Ordering::SeqCst), 1);
        assert_eq!(poll(&mut done, &second_waker), Poll::Ready(()));
    }

    #[test]
    fn task_done_wakes_multiple_waiters_and_is_ready_after_completion() {
        let state = GuestTaskHandleState::new();
        let first_counter = Arc::new(WakeCounter::default());
        let second_counter = Arc::new(WakeCounter::default());
        let first_waker = waker(&first_counter);
        let second_waker = waker(&second_counter);
        let mut first = TaskDone::new(StoreId::allocate(), state.clone());
        let mut second = TaskDone::new(StoreId::allocate(), state.clone());

        assert_eq!(poll(&mut first, &first_waker), Poll::Pending);
        assert_eq!(poll(&mut second, &second_waker), Poll::Pending);
        assert_eq!(active_waiters(&state), 2);

        state.complete();

        assert_eq!(first_counter.wakes.load(Ordering::SeqCst), 1);
        assert_eq!(second_counter.wakes.load(Ordering::SeqCst), 1);
        assert_eq!(poll(&mut first, &first_waker), Poll::Ready(()));
        assert_eq!(poll(&mut second, &second_waker), Poll::Ready(()));

        let third_counter = Arc::new(WakeCounter::default());
        let third_waker = waker(&third_counter);
        let mut third = TaskDone::new(StoreId::allocate(), state);
        assert_eq!(poll(&mut third, &third_waker), Poll::Ready(()));
        assert_eq!(third_counter.wakes.load(Ordering::SeqCst), 0);
    }
}
