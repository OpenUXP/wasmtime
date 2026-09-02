use crate::component::concurrent::TaskId;
use crate::component::concurrent::{self, GuestTaskHandle, GuestTaskId, PreparedCall};
use crate::component::func::LowerContext;
use crate::component::{AsAccessor, ComponentNamedList, Func, Lift, Lower, TypedFunc, Val};
use crate::prelude::*;
use crate::runtime::vm::SendSyncPtr;
use crate::{AsContextMut, StoreContextMut, ValRaw};
use core::marker;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use wasmtime_environ::component::{InterfaceType, MAX_FLAT_PARAMS, MAX_FLAT_RESULTS};

/// Returned from [`Func::start_call_concurrent`] to represent a
/// pending-but-not-yet-resolved call into wasm.
pub struct FuncCallConcurrent<'a, T> {
    call: concurrent::StagedCall<Vec<Val>>,
    results: &'a mut [Val],
    _marker: marker::PhantomData<fn(T)>,
}

impl Func {
    /// Start a concurrent call to this function.
    ///
    /// Concurrency is achieved by relying on the [`Accessor`] argument, which
    /// can be obtained by calling [`StoreContextMut::run_concurrent`].
    ///
    /// Unlike [`Self::call`] and [`Self::call_async`] (both of which require
    /// exclusive access to the store until the completion of the call), calls
    /// made using this method may run concurrently with other calls to the same
    /// instance.  In addition, the runtime will call the `post-return` function
    /// (if any) automatically when the guest task completes.
    ///
    /// # Progress
    ///
    /// For the wasm task being created in `call_concurrent` to make progress it
    /// must be run within the scope of [`run_concurrent`]. If there are no
    /// active calls to [`run_concurrent`] then the wasm task will appear as
    /// stalled. This is typically not a concern as an [`Accessor`] is bound
    /// by default to a scope of [`run_concurrent`].
    ///
    /// One situation in which this can arise, for example, is that if a
    /// [`run_concurrent`] computation finishes its async closure before all
    /// wasm tasks have completed, then there will be no scope of
    /// [`run_concurrent`] anywhere. In this situation the wasm tasks that have
    /// not yet completed will not make progress until [`run_concurrent`] is
    /// called again.
    ///
    /// Embedders will need to ensure that this future is `await`'d within the
    /// scope of [`run_concurrent`] to ensure that the value can be produced
    /// during the `await` call.
    ///
    /// # Cancellation
    ///
    /// To request cancellation, use [`Self::start_call_concurrent`] to obtain
    /// a [`FuncCallConcurrent`], call [`FuncCallConcurrent::task_handle`], and
    /// then call [`GuestTaskHandle::cancel`].
    ///
    /// If parameter lowering has not begun, the guest is not entered and the
    /// call-result future returns an error which can be downcast to
    /// [`GuestTaskCancelled`](crate::component::GuestTaskCancelled). Once
    /// parameter lowering begins, cancellation is cooperative and asynchronous:
    /// the guest may take an arbitrary amount of time to observe the request,
    /// may ignore it, or may call `task.return` instead of `task.cancel`. If the
    /// guest acknowledges cancellation with `task.cancel`, the same typed error
    /// is returned. If the guest calls `task.return`, the result is returned
    /// normally.
    ///
    /// A guest task's implicit thread may continue running after producing a
    /// result. Use [`GuestTaskHandle::task_done`] to wait until that thread has
    /// exited. Explicit threads created by the guest are outside this
    /// completion condition. Hard cancellation of an individual task is not
    /// supported; dropping the entire store remains the only way to forcibly
    /// stop a non-cooperating task.
    ///
    /// This async function behaves more like a "spawn" than a normal Rust async
    /// function. Dropping the returned future does not cancel the in-progress
    /// guest task; it only relinquishes the host's ability to observe the call
    /// result.
    ///
    /// This function will return an error if [`Config::concurrency_support`] is
    /// disabled.
    ///
    /// [`Config::concurrency_support`]: crate::Config::concurrency_support
    /// [`run_concurrent`]: crate::Store::run_concurrent
    /// [`Accessor`]: crate::component::Accessor
    ///
    /// # Panics
    ///
    /// Panics if the store that the [`Accessor`] is derived from does not own
    /// this function.
    ///
    /// # Example
    ///
    /// Using [`StoreContextMut::run_concurrent`] to get an [`Accessor`]:
    ///
    /// ```
    /// # use {
    /// #   wasmtime::{
    /// #     error::{Result},
    /// #     component::{Component, Linker, ResourceTable},
    /// #     Config, Engine, Store
    /// #   },
    /// # };
    /// #
    /// # struct Ctx { table: ResourceTable }
    /// #
    /// # async fn foo() -> Result<()> {
    /// # let mut config = Config::new();
    /// # let engine = Engine::new(&config)?;
    /// # let mut store = Store::new(&engine, Ctx { table: ResourceTable::new() });
    /// # let mut linker = Linker::new(&engine);
    /// # let component = Component::new(&engine, "")?;
    /// # let instance = linker.instantiate_async(&mut store, &component).await?;
    /// let my_func = instance.get_func(&mut store, "my_func").unwrap();
    /// store.run_concurrent(async |accessor| -> wasmtime::Result<_> {
    ///    my_func.call_concurrent(accessor, &[], &mut Vec::new()).await?;
    ///    Ok(())
    /// }).await??;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn call_concurrent(
        self,
        accessor: impl AsAccessor<Data: Send>,
        params: &[Val],
        results: &mut [Val],
    ) -> Result<()> {
        let accessor = accessor.as_accessor();
        let call = accessor.with(|store| self.start_call_concurrent(store, params, results))?;
        self.finish_call_concurrent(accessor, call).await
    }

    /// Performs preparatory work for invoking this function with `params`,
    /// returning a [`FuncCallConcurrent`]
    /// which can be passed to [`Func::finish_call_concurrent`] to resolve
    /// the call.
    ///
    /// For more information see [`Func::call_concurrent`].
    pub fn start_call_concurrent<'a, T: Send + 'static>(
        self,
        mut store: impl AsContextMut<Data = T>,
        params: &'a [Val],
        results: &'a mut [Val],
    ) -> Result<FuncCallConcurrent<'a, T>> {
        self.check_params_results(store.as_context_mut(), params, results)?;
        let prepared = self.prepare_call_dynamic(store.as_context_mut(), params.to_vec())?;
        let call = concurrent::StagedCall::new(store.as_context_mut(), prepared)?;
        Ok(FuncCallConcurrent {
            call,
            results,
            _marker: marker::PhantomData,
        })
    }

    /// Completes a call that was initiated via
    /// [`Func::start_call_concurrent`].
    pub async fn finish_call_concurrent<T: Send>(
        self,
        accessor: impl AsAccessor<Data = T>,
        call: FuncCallConcurrent<'_, T>,
    ) -> Result<()> {
        // Intentionally not used today, but left here for future API
        // compatibility with using this.
        let _ = accessor;
        let FuncCallConcurrent { call, results, .. } = call;
        let run_results = call.await?;
        assert_eq!(run_results.len(), results.len());
        for (result, slot) in run_results.into_iter().zip(results) {
            *slot = result;
        }
        Ok(())
    }

    /// Calls `concurrent::prepare_call` with monomorphized functions for
    /// lowering the parameters and lifting the result.
    fn prepare_call_dynamic<'a, T: Send + 'static>(
        self,
        mut store: StoreContextMut<'a, T>,
        params: Vec<Val>,
    ) -> Result<PreparedCall<Vec<Val>>> {
        let store = store.as_context_mut();
        let (options, flags, ty, raw_options) = self.abi_info(store.0);
        let async_ = raw_options.async_;
        let instance = self.instance();

        concurrent::prepare_call(
            store,
            self,
            MAX_FLAT_PARAMS,
            false,
            move |store, params_out| {
                Func::with_lower_context(instance, store, options, flags, ty, |cx, ty| {
                    Self::lower_args(cx, &params, ty, params_out)
                })
            },
            move |store, results| {
                let max_flat = if async_ {
                    MAX_FLAT_PARAMS
                } else {
                    MAX_FLAT_RESULTS
                };
                let results = Func::with_lift_context(instance, store, options, ty, |cx, ty| {
                    Self::lift_results(cx, ty, results, max_flat)?.collect::<Result<Vec<_>>>()
                })?;
                Ok(Box::new(results))
            },
        )
    }
}

impl<T> FuncCallConcurrent<'_, T> {
    /// Returns the diagnostic identifier for the task represented by this call.
    ///
    /// This can be later correlated with [`StoreContextMut::async_call_stack`]
    /// for example.
    pub fn task(&self) -> GuestTaskId {
        self.call.task()
    }

    /// Returns a handle which may be used to request cancellation with
    /// [`GuestTaskHandle::cancel`] or wait for the call's implicit thread to
    /// exit with [`GuestTaskHandle::task_done`].
    pub fn task_handle(&self) -> GuestTaskHandle {
        self.call.task_handle()
    }
}

/// Returned from [`TypedFunc::start_call_concurrent`] to represent a
/// pending-but-not-yet-resolved call into wasm.
pub struct TypedFuncCallConcurrent<T, P, R> {
    call: concurrent::StagedCall<R>,
    _marker: marker::PhantomData<fn(T, P)>,
}

impl<Params, Return> TypedFunc<Params, Return>
where
    Params: ComponentNamedList + Lower,
    Return: ComponentNamedList + Lift,
{
    pub(crate) async fn call_async_concurrent(
        &self,
        mut store: impl AsContextMut<Data: Send>,
        params: Params,
    ) -> Result<Return>
    where
        Return: 'static,
    {
        let mut store = store.as_context_mut();
        let ptr = SendSyncPtr::from(NonNull::from(&params).cast::<u8>());
        let prepared = self.prepare_call(store.as_context_mut(), true, move |cx, ty, dst| {
            // SAFETY: The goal here is to get `Params`, a non-`'static`
            // value, to live long enough to the lowering of the
            // parameters. We're guaranteed that `Params` lives in the
            // future of the outer function (we're in an `async fn`) so it'll
            // stay alive as long as the future itself. That is distinct,
            // for example, from the signature of `call_concurrent` below.
            //
            // Here a pointer to `Params` is smuggled to this location
            // through a `SendSyncPtr<u8>` to thwart the `'static` check
            // of rustc and the signature of `prepare_call`.
            //
            // Note the use of `SignalOnDrop` in the code that follows
            // this closure, which ensures that the task will be removed
            // from the concurrent state to which it belongs when the
            // containing `Future` is dropped, so long as the parameters
            // have not yet been lowered. Since this closure is removed from
            // the task after the parameters are lowered, it will never be called
            // after the containing `Future` is dropped.
            let params = unsafe { ptr.cast::<Params>().as_ref() };
            Self::lower_args(cx, ty, dst, params)
        })?;

        struct SignalOnDrop<'a, T: 'static> {
            store: StoreContextMut<'a, T>,
            task: TaskId,
        }

        impl<'a, T> Drop for SignalOnDrop<'a, T> {
            fn drop(&mut self) {
                self.task.host_future_dropped(self.store.0).unwrap();
            }
        }

        let mut wrapper = SignalOnDrop {
            store,
            task: prepared.task_id(),
        };

        let result = concurrent::StagedCall::new(wrapper.store.as_context_mut(), prepared)?;
        wrapper
            .store
            .as_context_mut()
            .run_concurrent_trap_on_idle(async |_| Ok(result.await?))
            .await?
    }

    /// Start a concurrent call to this function.
    ///
    /// Concurrency is achieved by relying on the [`Accessor`] argument, which
    /// can be obtained by calling [`StoreContextMut::run_concurrent`].
    ///
    /// Unlike [`Self::call`] and [`Self::call_async`] (both of which require
    /// exclusive access to the store until the completion of the call), calls
    /// made using this method may run concurrently with other calls to the same
    /// instance.  In addition, the runtime will call the `post-return` function
    /// (if any) automatically when the guest task completes.
    ///
    /// This function will return an error if [`Config::concurrency_support`] is
    /// disabled.
    ///
    /// [`Config::concurrency_support`]: crate::Config::concurrency_support
    ///
    /// # Progress and Cancellation
    ///
    /// For more information about how to make progress on the wasm task or how
    /// to cancel the wasm task see the documentation for
    /// [`Func::call_concurrent`].
    ///
    /// [`Func::call_concurrent`]: crate::component::Func::call_concurrent
    ///
    /// # Panics
    ///
    /// Panics if the store that the [`Accessor`] is derived from does not own
    /// this function.
    ///
    /// [`Accessor`]: crate::component::Accessor
    ///
    /// # Example
    ///
    /// Using [`StoreContextMut::run_concurrent`] to get an [`Accessor`]:
    ///
    /// ```
    /// # use {
    /// #   wasmtime::{
    /// #     error::{Result},
    /// #     component::{Component, Linker, ResourceTable},
    /// #     Config, Engine, Store
    /// #   },
    /// # };
    /// #
    /// # struct Ctx { table: ResourceTable }
    /// #
    /// # async fn foo() -> Result<()> {
    /// # let mut config = Config::new();
    /// # let engine = Engine::new(&config)?;
    /// # let mut store = Store::new(&engine, Ctx { table: ResourceTable::new() });
    /// # let mut linker = Linker::new(&engine);
    /// # let component = Component::new(&engine, "")?;
    /// # let instance = linker.instantiate_async(&mut store, &component).await?;
    /// let my_typed_func = instance.get_typed_func::<(), ()>(&mut store, "my_typed_func")?;
    /// store.run_concurrent(async |accessor| -> wasmtime::Result<_> {
    ///    my_typed_func.call_concurrent(accessor, ()).await?;
    ///    Ok(())
    /// }).await??;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn call_concurrent(
        self,
        accessor: impl AsAccessor<Data: Send>,
        params: Params,
    ) -> Result<Return>
    where
        Params: 'static,
        Return: 'static,
    {
        let call = accessor
            .as_accessor()
            .with(|store| self.start_call_concurrent(store, params))?;
        self.finish_call_concurrent(accessor, call).await
    }

    /// Performs preparatory work for invoking this function with `params`,
    /// returning a [`TypedFuncCallConcurrent`]
    /// which can be passed to [`TypedFunc::finish_call_concurrent`] to resolve
    /// the call.
    ///
    /// For more information see [`TypedFunc::call_concurrent`].
    pub fn start_call_concurrent<T>(
        self,
        mut store: impl AsContextMut<Data = T>,
        params: Params,
    ) -> Result<TypedFuncCallConcurrent<T, Params, Return>>
    where
        T: Send + 'static,
        Params: 'static,
        Return: 'static,
    {
        let mut store = store.as_context_mut();
        let mut store = store.as_context_mut();
        ensure!(
            store.0.concurrency_support(),
            "cannot use `call_concurrent` Config::concurrency_support disabled",
        );

        let prepared = self.prepare_call(store.as_context_mut(), false, move |cx, ty, dst| {
            Self::lower_args(cx, ty, dst, &params)
        })?;
        let call = concurrent::StagedCall::new(store, prepared)?;
        Ok(TypedFuncCallConcurrent {
            call,
            _marker: marker::PhantomData,
        })
    }

    /// Completes a call that was initiated via
    /// [`TypedFunc::start_call_concurrent`].
    pub async fn finish_call_concurrent<T>(
        self,
        accessor: impl AsAccessor<Data = T>,
        call: TypedFuncCallConcurrent<T, Params, Return>,
    ) -> Result<Return>
    where
        T: Send + 'static,
        Params: 'static,
        Return: 'static,
    {
        // This is intentionally part of the public API but not used yet.
        // This'll likely want to be used in future refactorings.
        let _ = accessor;
        call.call.await
    }

    /// Calls `concurrent::prepare_call` with monomorphized functions for
    /// lowering the parameters and lifting the result according to the number
    /// of core Wasm parameters and results in the signature of the function to
    /// be called.
    fn prepare_call<T>(
        self,
        store: StoreContextMut<'_, T>,
        host_future_present: bool,
        lower: impl FnOnce(
            &mut LowerContext<T>,
            InterfaceType,
            &mut [MaybeUninit<ValRaw>],
        ) -> Result<()>
        + Send
        + Sync
        + 'static,
    ) -> Result<PreparedCall<Return>>
    where
        Return: 'static,
    {
        use crate::component::storage::slice_to_storage;
        debug_assert!(store.0.concurrency_support());

        let param_count = if Params::flatten_count() <= MAX_FLAT_PARAMS {
            Params::flatten_count()
        } else {
            1
        };
        let (options, flags, ty, raw_options) = self.func().abi_info(store.0);
        let instance = self.func().instance();
        let max_results = if raw_options.async_ {
            MAX_FLAT_PARAMS
        } else {
            MAX_FLAT_RESULTS
        };

        concurrent::prepare_call(
            store,
            *self.func(),
            param_count,
            host_future_present,
            move |store, params_out| {
                Func::with_lower_context(instance, store, options, flags, ty, |cx, ty| {
                    lower(cx, ty, params_out)
                })
            },
            move |store, results| {
                let result = if Return::flatten_count() <= max_results {
                    Func::with_lift_context(instance, store, options, ty, |cx, ty| {
                        // SAFETY: Per the safety requirements documented for the
                        // `ComponentType` trait, `Return::Lower` must be
                        // compatible at the binary level with a `[ValRaw; N]`,
                        // where `N` is `mem::size_of::<Return::Lower>() /
                        // mem::size_of::<ValRaw>()`.  And since this function
                        // is only used when `Return::flatten_count() <=
                        // MAX_FLAT_RESULTS` and `MAX_FLAT_RESULTS == 1`, `N`
                        // can only either be 0 or 1.
                        //
                        // See `ComponentInstance::exit_call` for where we use
                        // the result count passed from
                        // `wasmtime_environ::fact::trampoline`-generated code
                        // to ensure the slice has the correct length, and also
                        // `concurrent::start_call` for where we conservatively
                        // use a slice length of 1 unconditionally.  Also note
                        // that, as of this writing `slice_to_storage`
                        // double-checks the slice length is sufficient.
                        let results: &Return::Lower = unsafe { slice_to_storage(results) };
                        Self::lift_stack_result(cx, ty, results)
                    })?
                } else {
                    Func::with_lift_context(instance, store, options, ty, |cx, ty| {
                        Self::lift_heap_result(cx, ty, &results[0])
                    })?
                };
                Ok(Box::new(result))
            },
        )
    }
}

impl<T, P, R> TypedFuncCallConcurrent<T, P, R> {
    /// Returns the diagnostic identifier for the task represented by this call.
    ///
    /// This can be later correlated with [`StoreContextMut::async_call_stack`]
    /// for example.
    pub fn task(&self) -> GuestTaskId {
        self.call.task()
    }

    /// Returns a handle which may be used to request cancellation with
    /// [`GuestTaskHandle::cancel`] or wait for the call's implicit thread to
    /// exit with [`GuestTaskHandle::task_done`].
    pub fn task_handle(&self) -> GuestTaskHandle {
        self.call.task_handle()
    }
}
