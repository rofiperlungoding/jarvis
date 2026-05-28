# Requirements Document

## Introduction

JARVIS is a Windows-first, voice-driven AI assistant inspired by the JARVIS persona from Iron Man. JARVIS supports natural spoken conversation, launches and controls Windows applications, performs web search, controls system media and hardware (volume, brightness, music playback), sends messages and email, manages reminders/alarms/timers, retrieves weather/news/calendar information, reads and summarizes local files, executes desktop automation and scripts, and maintains long-term memory of user preferences and prior conversations. JARVIS exposes a witty, formal, and slightly sarcastic personality.

The system is designed around a real-time voice pipeline (wake-word detection, speech-to-text, language model reasoning with tool/function calling, and text-to-speech) with an extensible skill plugin architecture. The architecture targets sub-second end-to-end latency for conversational turns and supports both cloud and local model backends so the user can trade off latency, cost, and privacy.

This document captures functional requirements (one section per mandatory capability), non-functional requirements (latency, privacy, extensibility, OS compatibility, credentials, authorization), and correctness properties intended to be validated through property-based testing.

## Glossary

- **JARVIS**: The complete AI assistant system, encompassing all subsystems described below.
- **Voice_Pipeline**: The end-to-end audio processing chain comprising Wake_Word_Detector, STT_Engine, Dialog_Manager, and TTS_Engine.
- **Wake_Word_Detector**: The subsystem that monitors microphone input for a configured activation phrase (default "Jarvis") and signals the rest of the pipeline to begin capturing user speech.
- **STT_Engine**: The Speech-to-Text subsystem that converts captured audio into a text Transcript.
- **TTS_Engine**: The Text-to-Speech subsystem that converts an Assistant_Response into spoken audio.
- **Dialog_Manager**: The subsystem that takes a Transcript plus conversation context, invokes the LLM_Backend, dispatches Tool_Calls, and produces an Assistant_Response.
- **LLM_Backend**: The configured large language model used by Dialog_Manager for reasoning and tool selection. Mistral AI la Plateforme API (e.g., mistral-large-latest, mistral-small-latest) is the default LLM_Backend; local Ollama-hosted Mistral models may serve as fallback.
- **Mistral_API_Endpoint**: The configured Mistral la Plateforme HTTPS endpoint (default: https://api.mistral.ai) used by Dialog_Manager for cloud LLM inference.
- **Mistral_Model_Id**: The specific Mistral model identifier supplied to Mistral_API_Endpoint requests (default: mistral-large-latest).
- **Transcript**: The textual output produced by STT_Engine for a single user utterance.
- **Assistant_Response**: The structured response produced by Dialog_Manager, containing spoken text and optional Tool_Calls.
- **Tool_Call**: A structured invocation of a Skill, identified by a name and a JSON argument object that conforms to the Skill's declared schema.
- **Skill**: A discrete capability plugin (e.g., LaunchAppSkill, WeatherSkill) that exposes one or more callable tools to the Dialog_Manager.
- **Skill_Registry**: The component responsible for loading, validating, and dispatching Skills.
- **Memory_Store**: The persistent store of conversational history, user preferences, and learned facts, backed by a vector database for semantic retrieval.
- **Memory_Record**: A single stored item in Memory_Store, comprising text content, embedding vector, timestamp, and metadata tags.
- **Reminder_Service**: The subsystem that schedules and triggers reminders, alarms, and timers.
- **Automation_Service**: The subsystem that executes desktop automation tasks and user-authorized scripts on Windows.
- **Credential_Store**: The encrypted storage that holds API keys, OAuth tokens, and user secrets, backed by Windows DPAPI / Credential Manager.
- **Authorization_Policy**: The configured ruleset that determines which actions require explicit user confirmation before execution.
- **Destructive_Action**: Any action that modifies, deletes, or transmits data outside the local machine, including but not limited to: sending email/messages, deleting files, executing arbitrary scripts, and uninstalling applications.
- **Latency_Budget**: The end-to-end target time from end-of-user-speech to start-of-assistant-audio, defined as 800 milliseconds.
- **Wake_Word_FAR**: Wake Word False Acceptance Rate, the rate at which Wake_Word_Detector triggers on audio that does not contain the wake phrase.
- **Wake_Word_FRR**: Wake Word False Rejection Rate, the rate at which Wake_Word_Detector fails to trigger on audio that does contain the wake phrase.
- **Intent_Parser**: The component that maps a Transcript or LLM output into a normalized Intent structure used for routing to Skills.
- **Intent**: A normalized, schema-validated representation of user intent, comprising an action name and a typed argument map.
- **Conversation_State**: The serializable state of a single conversation, including message history, active tool invocations, and slot-filling progress.

## Requirements

### Requirement 1: Natural Voice Conversation

**User Story:** As a user, I want to hold spoken conversations with JARVIS covering small talk, factual questions, and follow-ups, so that I can interact hands-free in a natural manner.

#### Acceptance Criteria

1. WHILE the application is running, THE Wake_Word_Detector SHALL continuously monitor the default microphone input for the configured wake phrase.
2. WHEN the Wake_Word_Detector detects the wake phrase, THE Voice_Pipeline SHALL begin capturing user speech within 200 milliseconds.
3. WHEN the user finishes speaking, as determined by a voice activity detector with a 700 millisecond trailing-silence threshold, THE STT_Engine SHALL produce a Transcript of the captured audio.
4. WHEN a Transcript is produced, THE Dialog_Manager SHALL generate an Assistant_Response using the LLM_Backend, supplying the current Conversation_State and retrieved Memory_Records as context.
5. WHEN an Assistant_Response is generated, THE TTS_Engine SHALL synthesize and begin playing the response audio.
6. WHILE a conversation is active, THE Dialog_Manager SHALL retain the prior turns of the current session in Conversation_State and use them to resolve pronouns and follow-up references.
7. WHERE the user has enabled barge-in, WHEN the user speaks while TTS_Engine is playing, THE Voice_Pipeline SHALL stop playback within 150 milliseconds and capture the new utterance.
8. IF the STT_Engine produces an empty Transcript or a Transcript with confidence below 0.4, THEN THE Dialog_Manager SHALL prompt the user to repeat the request rather than invoking the LLM_Backend.

### Requirement 2: Application Launching

**User Story:** As a user, I want JARVIS to open Windows applications by voice, so that I can launch tools without using the keyboard or mouse.

#### Acceptance Criteria

1. THE Skill_Registry SHALL expose a LaunchAppSkill tool whose argument schema requires a single string field named "application".
2. WHEN the Dialog_Manager issues a Tool_Call to LaunchAppSkill with a value of "application" that resolves to a known executable in the application registry, THE Automation_Service SHALL start the application as a new process.
3. THE application registry SHALL include built-in entries for Google Chrome, Visual Studio Code, and Spotify, and SHALL support user-defined entries mapping spoken names to executable paths or URI handlers.
4. IF the requested "application" value does not match any entry in the application registry, THEN THE Automation_Service SHALL return an error result, AND THE Dialog_Manager SHALL ask the user to clarify or register the application.
5. WHEN an application is launched successfully, THE Dialog_Manager SHALL emit an Assistant_Response confirming the action by application name.

### Requirement 3: Web Search

**User Story:** As a user, I want JARVIS to search the web and return summarized answers, so that I can get information on topics outside the model's training data.

#### Acceptance Criteria

1. THE Skill_Registry SHALL expose a WebSearchSkill tool whose argument schema requires a string field "query" and an optional integer field "max_results" with a default of 5 and a maximum of 10.
2. WHEN the Dialog_Manager issues a Tool_Call to WebSearchSkill, THE Automation_Service SHALL submit "query" to the configured search provider and return the top "max_results" results, each containing title, URL, and snippet.
3. WHEN search results are returned, THE Dialog_Manager SHALL produce a spoken summary of the top results and SHALL cite source URLs in the textual transcript shown in the UI log.
4. IF the search provider returns zero results or an error, THEN THE Dialog_Manager SHALL inform the user that no results were found and SHALL offer to refine the query.

### Requirement 4: Music, Volume, and Brightness Control

**User Story:** As a user, I want JARVIS to control music playback, system volume, and screen brightness by voice, so that I can adjust my environment hands-free.

#### Acceptance Criteria

1. THE Skill_Registry SHALL expose a MediaControlSkill tool with an "action" field constrained to the set {play, pause, next, previous, stop}.
2. WHEN the Dialog_Manager issues a Tool_Call to MediaControlSkill, THE Automation_Service SHALL send the corresponding Windows media key event to the operating system.
3. THE Skill_Registry SHALL expose a VolumeSkill tool with an "operation" field constrained to {set, increase, decrease, mute, unmute} and an optional integer "level" field in the range 0 to 100.
4. WHEN the Dialog_Manager issues a Tool_Call to VolumeSkill with operation "set", THE Automation_Service SHALL set the system master output volume to "level" percent.
5. WHEN the Dialog_Manager issues a Tool_Call to VolumeSkill with operation "increase" or "decrease", THE Automation_Service SHALL adjust the system master output volume by the supplied "level", or by 10 percent if "level" is omitted.
6. THE Skill_Registry SHALL expose a BrightnessSkill tool with an "operation" field constrained to {set, increase, decrease} and an optional integer "level" field in the range 0 to 100.
7. WHEN the Dialog_Manager issues a Tool_Call to BrightnessSkill, THE Automation_Service SHALL adjust the primary display brightness via the Windows WMI MonitorBrightnessMethods interface.
8. IF the active display does not support programmatic brightness control, THEN THE Automation_Service SHALL return a "not_supported" error and THE Dialog_Manager SHALL inform the user of the limitation.

### Requirement 5: Email and Messaging

**User Story:** As a user, I want JARVIS to send email and chat messages by voice, so that I can communicate without typing.

#### Acceptance Criteria

1. THE Skill_Registry SHALL expose a SendEmailSkill tool whose argument schema requires "recipient", "subject", and "body" string fields.
2. WHEN the Dialog_Manager issues a Tool_Call to SendEmailSkill, THE Automation_Service SHALL read back the resolved recipient address, subject, and body to the user and SHALL require an explicit verbal confirmation before transmission, in accordance with the Authorization_Policy for Destructive_Action.
3. WHEN the user confirms transmission, THE Automation_Service SHALL send the email via the configured SMTP or provider API using credentials retrieved from the Credential_Store.
4. THE Skill_Registry SHALL expose a SendMessageSkill tool whose argument schema requires "channel", "recipient", and "body" string fields, where "channel" identifies a configured messaging provider.
5. WHEN the Dialog_Manager issues a Tool_Call to SendMessageSkill, THE Automation_Service SHALL apply the same confirmation flow specified for SendEmailSkill before transmission.
6. IF credentials for the requested provider are not present in the Credential_Store, THEN THE Automation_Service SHALL return a "missing_credentials" error and THE Dialog_Manager SHALL guide the user through credential setup.

### Requirement 6: Reminders, Alarms, and Timers

**User Story:** As a user, I want JARVIS to set reminders, alarms, and countdown timers, so that I am notified about future events without external apps.

#### Acceptance Criteria

1. THE Skill_Registry SHALL expose a ReminderSkill tool whose argument schema requires a "label" string field and a "trigger_at" ISO-8601 timestamp field.
2. WHEN the Dialog_Manager issues a Tool_Call to ReminderSkill, THE Reminder_Service SHALL persist a reminder record in local storage and SHALL schedule a trigger at "trigger_at".
3. THE Skill_Registry SHALL expose a TimerSkill tool whose argument schema requires an integer "duration_seconds" field greater than zero and an optional "label" string field.
4. WHEN the Dialog_Manager issues a Tool_Call to TimerSkill, THE Reminder_Service SHALL start a countdown of "duration_seconds" and SHALL trigger a notification when the countdown reaches zero.
5. WHEN a reminder, alarm, or timer triggers, THE Reminder_Service SHALL emit a Windows toast notification AND THE TTS_Engine SHALL announce the "label" if the user is currently engaged in or has just completed a conversation.
6. WHILE the application is not running, IF a scheduled reminder time is reached, THEN THE Reminder_Service SHALL trigger the reminder on the next application start within 30 seconds of launch.
7. THE Skill_Registry SHALL expose a ListReminderSkill tool that returns all pending reminders, alarms, and timers.

### Requirement 7: Weather, News, and Calendar

**User Story:** As a user, I want JARVIS to report weather, news headlines, and my calendar events on request, so that I can plan my day.

#### Acceptance Criteria

1. THE Skill_Registry SHALL expose a WeatherSkill tool whose argument schema accepts an optional "location" string field defaulting to the user's configured home location.
2. WHEN the Dialog_Manager issues a Tool_Call to WeatherSkill, THE Automation_Service SHALL retrieve current conditions and a 24-hour forecast for "location" from the configured weather provider.
3. THE Skill_Registry SHALL expose a NewsSkill tool whose argument schema accepts optional "topic" and "max_items" fields, where "max_items" defaults to 5 and is capped at 10.
4. WHEN the Dialog_Manager issues a Tool_Call to NewsSkill, THE Automation_Service SHALL retrieve the top headlines for "topic" from the configured news provider.
5. THE Skill_Registry SHALL expose a CalendarSkill tool with operations constrained to {list_today, list_range, create_event}, where list_range requires "start" and "end" ISO-8601 fields and create_event requires "title", "start", and "end" fields.
6. WHEN the Dialog_Manager issues a Tool_Call to CalendarSkill with operation "create_event", THE Automation_Service SHALL apply the confirmation flow specified for Destructive_Action before writing to the calendar provider.
7. IF a configured external provider for weather, news, or calendar returns an error or times out after 5 seconds, THEN THE Automation_Service SHALL return a "provider_unavailable" error and THE Dialog_Manager SHALL inform the user.

### Requirement 8: File Reading and Summarization

**User Story:** As a user, I want JARVIS to read and summarize files on my computer, so that I can understand documents without opening them.

#### Acceptance Criteria

1. THE Skill_Registry SHALL expose a ReadFileSkill tool whose argument schema requires an absolute "path" string field.
2. WHEN the Dialog_Manager issues a Tool_Call to ReadFileSkill, THE Automation_Service SHALL verify "path" resolves within a user-configured allowed-directory list and SHALL read the file contents.
3. THE Skill_Registry SHALL expose a SummarizeFileSkill tool whose argument schema requires a "path" string field and accepts an optional "max_words" integer field defaulting to 200.
4. WHEN the Dialog_Manager issues a Tool_Call to SummarizeFileSkill, THE Dialog_Manager SHALL invoke the LLM_Backend with the file contents and a summarization prompt and SHALL return a summary not exceeding "max_words" words.
5. THE Automation_Service SHALL support reading text-based formats including .txt, .md, .csv, .json, .py, .js, .ts, .pdf, .docx.
6. IF "path" resolves outside the allowed-directory list, THEN THE Automation_Service SHALL return an "access_denied" error and SHALL NOT read the file.
7. IF the file size exceeds 5 megabytes, THEN THE Automation_Service SHALL return a "file_too_large" error and THE Dialog_Manager SHALL offer to read a specific page or section instead.

### Requirement 9: Script Execution and Desktop Automation

**User Story:** As a user, I want JARVIS to run scripts and automate desktop interactions, so that I can offload routine tasks.

#### Acceptance Criteria

1. THE Skill_Registry SHALL expose a RunScriptSkill tool whose argument schema requires a "script_id" string field referencing a script registered in the user's script catalog.
2. WHEN the Dialog_Manager issues a Tool_Call to RunScriptSkill, THE Automation_Service SHALL look up "script_id" in the script catalog and, IF found, SHALL apply the confirmation flow specified for Destructive_Action before execution.
3. WHEN the user confirms execution, THE Automation_Service SHALL execute the script using the interpreter declared in the catalog entry (PowerShell, Python, or Batch) and SHALL capture stdout, stderr, and exit code.
4. IF "script_id" does not resolve to a catalog entry, THEN THE Automation_Service SHALL return a "script_not_found" error.
5. THE Automation_Service SHALL NOT execute arbitrary script text supplied directly in a Tool_Call argument; only registered "script_id" values are accepted.
6. THE Skill_Registry SHALL expose a DesktopAutomationSkill tool whose argument schema requires an "action" field constrained to a documented set of UI actions (click, type, hotkey, focus_window) and corresponding typed payload fields.
7. WHEN the Dialog_Manager issues a Tool_Call to DesktopAutomationSkill, THE Automation_Service SHALL perform the action via pyautogui or pywinauto and SHALL return a structured success or error result.
8. WHEN a script execution exceeds 60 seconds, THE Automation_Service SHALL terminate the process and SHALL return a "timeout" error.

### Requirement 10: Long-Term Memory

**User Story:** As a user, I want JARVIS to remember past conversations and my preferences, so that future interactions feel personalized and context-aware.

#### Acceptance Criteria

1. WHEN a conversation turn completes, THE Dialog_Manager SHALL persist a Memory_Record containing the user utterance, the Assistant_Response, a timestamp, and provenance metadata.
2. WHEN the user states a personal preference or fact (for example, a favorite music genre or a home location), THE Dialog_Manager SHALL extract and store the preference as a typed Memory_Record with category metadata.
3. WHEN the Dialog_Manager begins composing an Assistant_Response, THE Dialog_Manager SHALL query Memory_Store for the top K Memory_Records most similar to the current Transcript, where K is configurable and defaults to 5.
4. WHEN retrieved Memory_Records are returned, THE Dialog_Manager SHALL include them in the LLM_Backend prompt under a clearly delimited "memory" section.
5. THE Skill_Registry SHALL expose a MemoryAdminSkill tool with operations constrained to {list, search, forget}, where forget requires a "record_id" field.
6. WHEN the user requests that a fact be forgotten, THE Dialog_Manager SHALL invoke MemoryAdminSkill with operation "forget" and the matching record id, and THE Memory_Store SHALL delete the record.
7. THE Memory_Store SHALL persist data on the local filesystem encrypted at rest using the user's Windows account credentials via DPAPI.
8. WHERE the user has enabled memory-redaction, WHEN a Memory_Record contains values that match configured PII patterns, THE Dialog_Manager SHALL replace those values with redaction tokens before storage.

### Requirement 11: JARVIS Personality

**User Story:** As a user, I want JARVIS to respond with a witty, formal, and slightly sarcastic tone, so that interactions feel like the cinematic JARVIS character.

#### Acceptance Criteria

1. THE Dialog_Manager SHALL prepend a system prompt to every LLM_Backend invocation that defines the JARVIS persona as witty, formal, and lightly sarcastic, addresses the user as "sir" or by configured honorific, and forbids breaking character.
2. THE Dialog_Manager SHALL configure the TTS_Engine with a voice profile selected to match the JARVIS persona (mature, calm, British-accented by default) where the configured TTS provider supports voice selection.
3. WHILE the user is in an active conversation, THE Dialog_Manager SHALL maintain consistent persona tone across turns by including the persona system prompt in every LLM_Backend invocation.
4. WHERE the user has configured a custom persona profile, THE Dialog_Manager SHALL apply that profile in place of the default JARVIS profile.
5. IF the LLM_Backend returns a response that violates the persona constraints (for example, refers to itself as "ChatGPT" or "Claude"), THEN THE Dialog_Manager SHALL rewrite or regenerate the response before forwarding it to the TTS_Engine.

### Requirement 12: Latency Budget (Non-Functional)

**User Story:** As a user, I want JARVIS to feel conversationally responsive, so that interactions do not feel laggy.

#### Acceptance Criteria

1. WHEN the user finishes speaking, THE Voice_Pipeline SHALL begin emitting TTS audio within the Latency_Budget of 800 milliseconds for at least 90 percent of conversational turns under typical home-network conditions.
2. THE Dialog_Manager SHALL stream LLM_Backend tokens to the TTS_Engine as they arrive, beginning TTS synthesis as soon as the first sentence boundary is reached.
3. WHILE Tool_Calls are executing, THE Dialog_Manager SHALL emit a configurable acknowledgement utterance ("One moment") via TTS_Engine if total tool execution time exceeds 1.5 seconds.
4. IF the Mistral_API_Endpoint is unreachable for more than 3 seconds OR returns 5xx errors, THEN THE Dialog_Manager SHALL fall back to the configured local LLM_Backend (Ollama-hosted Mistral or compatible model) if available, AND SHALL inform the user of the fallback.

### Requirement 13: Privacy and Security (Non-Functional)

**User Story:** As a user, I want my voice data, credentials, and personal information protected, so that I can trust JARVIS with private content.

#### Acceptance Criteria

1. THE Credential_Store SHALL persist all third-party API keys and OAuth tokens encrypted via Windows DPAPI scoped to the current user account.
2. THE Voice_Pipeline SHALL NOT transmit raw audio to any cloud service WHERE the user has selected a local STT_Engine in configuration.
3. WHERE the user enables incognito mode, WHILE incognito mode is active, THE Dialog_Manager SHALL NOT persist any Memory_Record from the current session.
4. THE JARVIS application SHALL log all outbound network destinations and the user-visible justification for each (for example, "openai.com: LLM completion") in a local audit log.
5. WHEN the user requests deletion of all stored data, THE Memory_Store, Credential_Store, and audit log SHALL be erased within 5 seconds of confirmation.
6. IF a Skill attempts to read a path outside the allowed-directory list or to invoke a network endpoint not in the configured allowlist, THEN THE Skill_Registry SHALL block the operation and SHALL log a "policy_violation" entry to the audit log.

### Requirement 14: Extensibility via Skill Plugins (Non-Functional)

**User Story:** As a developer or power user, I want to add new skills without modifying core code, so that JARVIS can be extended for personal workflows.

#### Acceptance Criteria

1. THE Skill_Registry SHALL discover Skills at startup by scanning a configured plugin directory for Python modules that declare a Skill class implementing the documented Skill interface.
2. THE Skill interface SHALL require each Skill to expose a name, a JSON Schema describing its tool argument structure, a human-readable description, and an executor function.
3. WHEN a Skill is loaded, THE Skill_Registry SHALL validate its JSON Schema against the Skill interface contract AND SHALL refuse to register the Skill if validation fails.
4. WHEN the Dialog_Manager invokes a Tool_Call, THE Skill_Registry SHALL validate the supplied argument object against the Skill's JSON Schema before dispatching execution.
5. IF a Tool_Call argument fails schema validation, THEN THE Skill_Registry SHALL return a "schema_violation" error WITHOUT executing the Skill, AND THE Dialog_Manager SHALL ask the LLM_Backend to retry with corrected arguments up to a maximum of 2 retries.
6. THE Skill_Registry SHALL support the Model Context Protocol (MCP) as an additional Skill source, allowing external MCP servers configured by the user to contribute tools.

### Requirement 15: Operating System Compatibility (Non-Functional)

**User Story:** As a user, I want JARVIS to run on Windows today and on macOS or Linux later, so that I am not locked into a single OS.

#### Acceptance Criteria

1. THE JARVIS application SHALL run on Windows 10 version 1903 and later AND on Windows 11.
2. THE Automation_Service SHALL isolate Windows-specific calls (for example, WMI brightness, DPAPI, Win32 media keys) behind a platform abstraction interface.
3. WHERE the platform abstraction interface is implemented for macOS or Linux, THE corresponding Skills SHALL function on those platforms without changes to the Dialog_Manager or Skill_Registry.
4. IF a Skill is invoked on a platform where its underlying capability is not implemented, THEN THE Automation_Service SHALL return a "platform_not_supported" error.

### Requirement 16: Authorization for Destructive Actions (Non-Functional)

**User Story:** As a user, I want JARVIS to confirm before taking irreversible or sensitive actions, so that misheard commands do not cause harm.

#### Acceptance Criteria

1. THE Authorization_Policy SHALL classify the following Tool_Calls as Destructive_Action: SendEmailSkill, SendMessageSkill, RunScriptSkill, CalendarSkill.create_event, MemoryAdminSkill.forget, and any Skill explicitly declaring `destructive: true` in its manifest.
2. WHEN the Dialog_Manager would dispatch a Tool_Call classified as Destructive_Action, THE Dialog_Manager SHALL produce a spoken summary of the intended action and SHALL require an affirmative user response before invoking the Skill.
3. WHERE the user has configured a trusted-action allowlist, WHEN the requested Tool_Call matches an entry in that allowlist with matching arguments, THE Dialog_Manager SHALL bypass confirmation for that single invocation.
4. WHEN the user denies confirmation, THE Dialog_Manager SHALL cancel the Tool_Call and SHALL acknowledge the cancellation.
5. THE JARVIS application SHALL record every Destructive_Action attempt, its confirmation outcome, and its execution result in the audit log.

### Requirement 17: Error Handling (Non-Functional)

**User Story:** As a user, I want JARVIS to recover gracefully from failures, so that a single error does not require restarting the application.

#### Acceptance Criteria

1. IF any Skill executor raises an unhandled exception, THEN THE Skill_Registry SHALL catch the exception, SHALL log the stack trace, AND SHALL return a structured error result to the Dialog_Manager.
2. WHEN the Dialog_Manager receives a structured error result, THE Dialog_Manager SHALL produce a user-facing Assistant_Response describing the failure in plain language and SHALL suggest a remediation when possible.
3. IF the STT_Engine, TTS_Engine, or LLM_Backend fails to respond within its configured timeout, THEN THE Voice_Pipeline SHALL emit an audible error tone, log the failure, and return to the wake-word listening state.
4. WHEN the JARVIS application crashes, THE next launch SHALL detect the prior crash and SHALL offer to submit an anonymized diagnostic report.

### Requirement 18: Wake Word Custom Configuration (Non-Functional)

**User Story:** As a user, I want to configure my own wake word, so that I can use a name other than the default.

#### Acceptance Criteria

1. WHERE the user has configured a custom wake phrase, THE Wake_Word_Detector SHALL load the corresponding model and SHALL detect the configured phrase in place of the default.
2. THE Wake_Word_Detector SHALL maintain Wake_Word_FAR at or below 0.5 false activations per hour averaged across a 24-hour validation corpus.
3. THE Wake_Word_Detector SHALL maintain Wake_Word_FRR at or below 0.05 across a validation set of 200 wake-phrase utterances captured at 1 to 3 meters from the microphone.

### Requirement 19: LLM Backend Configuration (Mistral AI)

**User Story:** As a user, I want JARVIS to use Mistral AI as its language model, so that I get high-quality reasoning with European data residency and competitive pricing.

#### Acceptance Criteria

1. THE Dialog_Manager SHALL use Mistral_API_Endpoint as the default LLM_Backend.
2. THE default Mistral_Model_Id SHALL be mistral-large-latest.
3. THE Dialog_Manager SHALL retrieve the Mistral API key from the Credential_Store at startup AND SHALL never write the key value to logs, the audit log, telemetry, or any persisted file outside the Credential_Store.
4. THE Dialog_Manager SHALL invoke Mistral function calling for Tool_Call dispatch, mapping each Skill's JSON Schema to a Mistral function definition.
5. THE Dialog_Manager SHALL stream responses via Mistral's streaming API and SHALL forward tokens to the TTS_Engine at sentence boundaries to enable progressive synthesis as specified in Requirement 12 acceptance criterion 2.
6. WHERE the user configures a different Mistral_Model_Id (for example mistral-small-latest, codestral-latest, or magistral-medium-latest), THE Dialog_Manager SHALL use that Mistral_Model_Id for subsequent LLM_Backend invocations.
7. IF the Mistral_API_Endpoint returns HTTP 401 or HTTP 403, THEN THE Dialog_Manager SHALL inform the user that the Mistral API key is invalid AND SHALL guide the user through Credential_Store update for the Mistral key.
8. IF the Mistral_API_Endpoint returns HTTP 429, THEN THE Dialog_Manager SHALL retry the request using exponential backoff with a maximum of 3 retries before surfacing a "rate_limited" error to the user.

## Correctness Properties

The following correctness properties are intended to be verified through property-based testing during implementation. Each property is paired with the requirement(s) it validates.

### CP1: Intent Parser Idempotency
For every Transcript T, parse_intent(T) SHALL be equal to parse_intent(serialize_intent(parse_intent(T))). That is, parsing an Intent and then serializing and re-parsing it SHALL yield the same Intent. Validates Requirement 1, Requirement 14.

### CP2: Tool Call Schema Validation Soundness
For every Skill S registered in Skill_Registry and for every argument object A, IF validate(A, S.schema) returns true, THEN dispatching the Tool_Call(S, A) SHALL NOT produce a "schema_violation" error. Conversely, IF validate(A, S.schema) returns false, THEN dispatching SHALL produce exactly a "schema_violation" error and SHALL NOT invoke the Skill executor. Validates Requirement 14.

### CP3: Memory Retrieval Determinism
For every Memory_Store snapshot M and every query Q, retrieve(M, Q, K) SHALL return the same ordered list of Memory_Records on every call within a single session, given the same K and the same embedding model version. Validates Requirement 10.

### CP4: Memory Forget Removes Record
For every Memory_Record R stored in Memory_Store M, after invoking MemoryAdminSkill.forget(R.id), every retrieve(M, Q, K) for any Q and K SHALL NOT include R in its results. Validates Requirement 10, Requirement 13.

### CP5: STT/TTS Round Trip Integrity (Coarse)
For every short text T drawn from a domain-representative corpus, the normalized form of STT_Engine(TTS_Engine(T)) SHALL match T under a defined edit-distance tolerance (Word Error Rate at most 0.10 for utterances under 8 words). Validates Requirement 1, Requirement 11.

### CP6: Conversation State Determinism
For every initial Conversation_State S0 and every sequence of inputs I1...In, applying the Dialog_Manager transition function with stubbed LLM_Backend (deterministic mock) and stubbed time source SHALL produce the same final Conversation_State on every run. Determinism is verified against stubbed Mistral responses, since the live Mistral_API_Endpoint is non-deterministic by design. Validates Requirement 1, Requirement 17, Requirement 19.

### CP7: Wake Word False Acceptance Rate Bound
Across a 24-hour validation audio corpus that does not contain the configured wake phrase, the count of Wake_Word_Detector activations divided by 24 SHALL be at most 0.5. Validates Requirement 18.

### CP8: Wake Word False Rejection Rate Bound
Across a validation set of N wake-phrase utterances (N at least 200), the fraction of utterances for which Wake_Word_Detector does not activate SHALL be at most 0.05. Validates Requirement 18.

### CP9: Authorization Policy Coverage
For every Tool_Call C dispatched by Dialog_Manager, IF C.skill is classified as Destructive_Action by Authorization_Policy AND C is not matched by the trusted-action allowlist, THEN there SHALL exist a corresponding "confirmation_requested" entry in the audit log immediately preceding the dispatch. Validates Requirement 16.

### CP10: Skill Plugin Isolation
For every Skill S that raises an exception during execution, the Dialog_Manager SHALL still produce an Assistant_Response and the Voice_Pipeline SHALL return to the wake-word listening state. That is, no Skill failure SHALL crash the Voice_Pipeline. Validates Requirement 17.

### CP11: Credential Store Confidentiality
For every secret value V written to Credential_Store, V SHALL NOT appear in plaintext in any file under the application data directory or in the audit log. Validates Requirement 13.

### CP12: Path Sandbox Soundness
For every "path" argument P supplied to ReadFileSkill or SummarizeFileSkill, IF P (after canonicalization, including resolution of symbolic links and "..") does not lie within the allowed-directory list, THEN the Skill SHALL return "access_denied" and SHALL NOT open the file. Validates Requirement 8, Requirement 13.

### CP13: Reminder Trigger Monotonicity
For every set of scheduled reminders with trigger times T1 < T2 < ... < Tn, when the system clock advances monotonically through those times, Reminder_Service SHALL fire the corresponding triggers in the same order T1, T2, ..., Tn. Validates Requirement 6.

### CP14: Persona Invariance
For every LLM_Backend invocation issued by Dialog_Manager, the rendered prompt SHALL contain the active persona system message as its first message. Validates Requirement 11.

### CP15: Mistral Tool Call Schema Conformance
For every Skill S registered in Skill_Registry, the JSON Schema produced by mapping S to a Mistral function definition SHALL conform to Mistral function-calling format constraints (object-typed parameters, JSON Schema draft-07 subset supported by Mistral, no unsupported keywords). For every generated Mistral function definition F, F SHALL round-trip through Mistral's function schema validator without modification. Validates Requirement 14, Requirement 19.
