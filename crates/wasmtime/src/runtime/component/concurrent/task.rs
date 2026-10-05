use super::{AsAccessor, GuestTaskId, TaskId};
use crate::store::StoreId;
use crate::try_mutex::TryMutex;
use crate::{AsContextMut, Result};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::future;
use core::mem;
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

/// Error returned by a concurrent call's result future when the host
/// cancels the task before parameter lowering begins or when the guest
/// acknowledges a cancellation request by calling the `task.cancel` intrinsic.
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
    Running(Vec<Waker>),
    Complete,
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

        future::poll_fn(|cx| {
            super::check_ambient_store(self.store);
            self.state.poll_complete(cx)
        })
        .await
    }
}

impl GuestTaskHandleState {
    pub(super) fn new() -> GuestTaskHandleState {
        GuestTaskHandleState {
            state: Arc::new(TryMutex::new(GuestTaskState::Running(Vec::new()))),
        }
    }

    pub(super) fn is_complete(&self) -> bool {
        matches!(
            &*self.state.try_lock().expect("should not be contended"),
            GuestTaskState::Complete
        )
    }

    fn poll_complete(&self, cx: &mut Context<'_>) -> Poll<()> {
        let mut state = self.state.try_lock().expect("should not be contended");
        match &mut *state {
            GuestTaskState::Running(waiters) => {
                if !waiters.iter().any(|waker| waker.will_wake(cx.waker())) {
                    waiters.push(cx.waker().clone());
                }
                Poll::Pending
            }
            GuestTaskState::Complete => Poll::Ready(()),
        }
    }

    pub(super) fn complete(&self) {
        let waiters = match mem::replace(
            &mut *self.state.try_lock().expect("should not be contended"),
            GuestTaskState::Complete,
        ) {
            GuestTaskState::Running(waiters) => waiters,
            GuestTaskState::Complete => return,
        };
        for waker in waiters {
            waker.wake();
        }
    }
}
