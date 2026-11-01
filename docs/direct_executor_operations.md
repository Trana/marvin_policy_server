# Policy backend lifecycle operations

Ticket: https://github.com/Trana/marvin_controller/issues/1

`prepare_execution(generation, initial_state_json)` and `request_activation(active)`
return plain `ExecutionOutcome` values. ROS callbacks translate requests and copy
success/message fields only. NativePolicyServer overrides these backend operations,
so native validation and acknowledgement no longer depend on ROS response objects.

Operations and inference still use the existing executor thread. The extraction
preserves activation validation, inference reset, generation observation, native
acknowledgements, deactivation history and fault handling order. No new timer or
thread is introduced. Execution-state output and service names are unchanged.

The supervisor keeps a RemotePolicyExecutor. Its remote timeout and output-release
logic cannot be replaced by a local activation boolean. Required physical tests
remain outstanding until separately authorized and completed.
