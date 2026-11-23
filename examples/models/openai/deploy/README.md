# OpenAI API in the robot station

OpenAIAPIClient uses the Responses API with the existing robot-use prompt and fenced JSON actions.
It sends named camera frames and measured state, keeps complete history, and replays output items including
encrypted reasoning with store=false. It imposes no application context limit and does not truncate history.
The API model's context window still applies. The model_context_window option is display metadata only
(default 1,050,000 for gpt-6-astra).

Configure the model alongside the existing local model:

```yaml
models:
  gpt_astra:
    label: GPT-6 Astra
    client_class: pear.deploy.station.agent_loop.clients.openai_api.OpenAIAPIClient
    timeout_s: 300
    client:
      base_url: https://api.openai.com/v1
      api_key_file: /path/outside/repository/openai_api_key
      model: gpt-6-astra
      reasoning_effort: xhigh
      max_output_tokens: 32000
```

The key file contains only the key and should have permissions 600. Omitting api_key_file uses OPENAI_API_KEY.
Keep credentials outside the repository and task logs. Camera images and history are sent to the provider.
The endpoint and model ID must match the account or gateway.

The health check verifies model access. Test Responses inference separately with saved snapshots:

```bash
python examples/models/openai/deploy/check_api.py \
  --config /path/to/station.yaml --model gpt_astra \
  --snapshot-dir /path/to/saved/cameras
```

This check expects files named by each configured RealSense SDK serial, with a .jpg extension.
It sends synthetic zero state and asks for zero changes, parses the complete answer through ActionParser,
and opens no robot hardware. It incurs an API request.

The configured i2rt station also offers GPT-6.2 Sol (key gpt_62_sol, model ID gpt-6.2-sol)
through the same gateway and key file. It uses xhigh reasoning, a 32,000-token per-turn output budget,
and the same complete-history behavior without an application context cap. Its model_context_window
of 1,050,000 is display metadata only.

When idle and motors disabled, restart the station GUI with the new configuration and choose
GPT-6 Astra or GPT-6.2 Sol from the model dropdown before creating a task. On the configured i2rt server:

```bash
/home/ubuntu/xuankun/pear-runtime/bin/run-pear-station
```

The default remains the local served model. No PEAR_MODEL environment variable is required for dropdown selection.

Only a completed final answer reaches ActionParser. Refusal, truncation, broken streams, API errors,
timeout and cancellation discard the pending history. Pausing cancels the local request; the agent loop
checks cancellation again before dispatch. Local cancellation does not guarantee that provider billing
stops immediately. There are no automatic API retries.

The page shows available reasoning summaries. Usage includes input, output, cached input and reasoning
tokens. Whole-request throughput and local decode-only throughput use different timing boundaries.

The station's existing task-loop and IK-retry behavior apply. This adapter adds no task-round limit.
Record interventions separately and verify object-placement success from video.

References: [Responses](https://developers.openai.com/api/reference/responses/overview),
[reasoning state](https://developers.openai.com/api/docs/guides/reasoning),
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra),
[GPT-6.2 Sol](https://developers.openai.com/api/docs/models/gpt-6.2-sol).


## Kimi K2.5 gateway

The Responses-compatible adapter was also tested with kimi-k2.5 on the configured
gateway: two turns of three synthetic camera images, complete history, and parsed
zero-delta actions. It correctly identified the synthetic shapes and colors.
Use a separate api_key_file and configure the provider's base_url, model=`kimi-k2.5`,
reasoning_effort=`high`, max_output_tokens=32000, and model_context_window=262144.
The context window is display metadata, not a local history cap; the gateway's
actual limit and interpretation of reasoning effort have not been independently verified.

deepseek-v4-pro text requests succeeded, but both Responses and Chat image probes
reported no visible images. Keep that entry out of the live robot dropdown until
visual input is verified. A valid zero action alone is not a vision test.
