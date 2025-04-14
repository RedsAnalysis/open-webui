<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { createEventDispatcher, onMount, getContext } from 'svelte';
	const dispatch = createEventDispatcher();

	import { getBackendConfig } from '$lib/apis';
	import {
		getAudioConfig,
		updateAudioConfig,
		getModels as _getModels,
		getVoices as _getVoices
	} from '$lib/apis/audio';
	import { config, settings } from '$lib/stores';

	import SensitiveInput from '$lib/components/common/SensitiveInput.svelte';

	import { TTS_RESPONSE_SPLIT } from '$lib/types';

	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	const i18n = getContext<Writable<i18nType>>('i18n');

	export let saveHandler: () => void;

	// Audio State Variables
	let TTS_OPENAI_API_BASE_URL = '';
	let TTS_OPENAI_API_KEY = '';
	// ADDED by RED: State variable for custom TTS API Base URL
	let TTS_CUSTOM_API_BASE_URL = '';
	let TTS_CUSTOM_API_KEY = '';

	let TTS_API_KEY = ''; // Used by ElevenLabs, Azure
	let TTS_ENGINE = '';
	let TTS_MODEL = '';
	let TTS_VOICE = '';
	let TTS_SPLIT_ON: TTS_RESPONSE_SPLIT = TTS_RESPONSE_SPLIT.PUNCTUATION;
	let TTS_AZURE_SPEECH_REGION = '';
	let TTS_AZURE_SPEECH_OUTPUT_FORMAT = '';

	let STT_OPENAI_API_BASE_URL = '';
	let STT_OPENAI_API_KEY = '';
	let STT_ENGINE = '';
	let STT_MODEL = '';
	let STT_WHISPER_MODEL = '';
	let STT_AZURE_API_KEY = '';
	let STT_AZURE_REGION = '';
	let STT_AZURE_LOCALES = '';
	let STT_DEEPGRAM_API_KEY = '';

	let STT_WHISPER_MODEL_LOADING = false;

	// Type definition for voices needs to accommodate both browser and backend APIs
	// eslint-disable-next-line no-undef
	let voices: (SpeechSynthesisVoice | { id: string; name: string; [key: string]: any })[] = [];
	let models: Awaited<ReturnType<typeof _getModels>>['models'] = [];

	// Function to fetch available TTS models based on the selected engine
	const getModels = async () => {
		// RESTORED by RED: Models are not fetched for Web API ('') or Transformers ('transformers')
		if (TTS_ENGINE === '' || TTS_ENGINE === 'transformers') {
			models = [];
		} else { // Fetch for OpenAI, ElevenLabs, Azure, Custom (assuming backend might provide)
			try {
				const res = await _getModels(
					localStorage.token,
					$config?.features?.enable_direct_connections && ($settings?.directConnections ?? null)
				);
				if (res) {
					console.log('Models response:', res);
					models = res.models || []; // Ensure it's an array
				} else {
					models = [];
				}
			} catch (e) {
				toast.error(`Error fetching models: ${e}`);
				models = []; // Reset on error
			}
		}
	};

	// Function to fetch available TTS voices based on the selected engine
	const getVoices = async () => {
		// Handle Web API voices (browser)
		if (TTS_ENGINE === '') {
			voices = []; // Clear previous voices first
			const getVoicesLoop = setInterval(() => {
				const browserVoices = speechSynthesis.getVoices();
				if (browserVoices.length > 0) {
					clearInterval(getVoicesLoop);
					voices = browserVoices.sort((a, b) =>
						a.name.localeCompare(b.name, $i18n.resolvedLanguage)
					);
				}
			}, 100);
		}
		// RESTORED by RED: Voices are not fetched for Transformers
		else if (TTS_ENGINE === 'transformers') {
			voices = [];
		}
		// Fetch for OpenAI, ElevenLabs, Azure, Custom (assuming backend might provide)
		else {
			voices = []; // Clear previous voices first
			try {
				const res = await _getVoices(localStorage.token);
				if (res) {
					console.log('Voices response:', res);
					voices = res.voices || []; // Ensure it's an array
					if (Array.isArray(voices)) {
						voices.sort((a, b) => a.name.localeCompare(b.name, $i18n.resolvedLanguage));
					}
				} else {
					voices = [];
				}
			} catch (e) {
				toast.error(`Error fetching voices: ${e}`);
				voices = []; // Reset on error
			}
		}
	};

	// Function to update the audio configuration on the backend
	const updateConfigHandler = async () => {
		console.log('Updating config with TTS Engine:', TTS_ENGINE);
		const res = await updateAudioConfig(localStorage.token, {
			tts: {
				OPENAI_API_BASE_URL: TTS_OPENAI_API_BASE_URL,
				OPENAI_API_KEY: TTS_OPENAI_API_KEY,

				// ADDED by RED: Pass custom URL to backend & Pass custom key to backend
				CUSTOMTTS_OPENAPI_BASE_URL: TTS_CUSTOM_API_BASE_URL,
				CUSTOMTTS_OPENAPI_KEY: TTS_CUSTOM_API_KEY,

				API_KEY: TTS_API_KEY, // Keep sending this for ElevenLabs/Azure
				ENGINE: TTS_ENGINE,
				MODEL: TTS_MODEL,
				VOICE: TTS_VOICE,
				SPLIT_ON: TTS_SPLIT_ON,
				AZURE_SPEECH_REGION: TTS_AZURE_SPEECH_REGION,
				AZURE_SPEECH_OUTPUT_FORMAT: TTS_AZURE_SPEECH_OUTPUT_FORMAT
			},
			stt: {
				OPENAI_API_BASE_URL: STT_OPENAI_API_BASE_URL,
				OPENAI_API_KEY: STT_OPENAI_API_KEY,
				ENGINE: STT_ENGINE,
				MODEL: STT_MODEL,
				WHISPER_MODEL: STT_WHISPER_MODEL,
				DEEPGRAM_API_KEY: STT_DEEPGRAM_API_KEY,
				AZURE_API_KEY: STT_AZURE_API_KEY,
				AZURE_REGION: STT_AZURE_REGION,
				AZURE_LOCALES: STT_AZURE_LOCALES
			}
		});

		if (res) {
			saveHandler(); // Call parent save handler (likely shows success toast)
			config.set(await getBackendConfig()); // Re-fetch global config
		}
	};

	// Handler for the local Whisper model update/download button
	const sttModelUpdateHandler = async () => {
		STT_WHISPER_MODEL_LOADING = true;
		await updateConfigHandler(); // Trigger save which might trigger download on backend
		STT_WHISPER_MODEL_LOADING = false;
	};

	// Handler for changes to the TTS engine dropdown
	// ADDED by RED: Specific handler for TTS engine change
	const handleTtsEngineChange = async (event: Event & { currentTarget: HTMLSelectElement }) => {
		const newEngine = event.currentTarget.value;
		TTS_ENGINE = newEngine; // Update state immediately

		
		// Fetch voices and models relevant to the newly selected engine
		// This will correctly result in empty arrays for '' and 'transformers'
		await getVoices();
		await getModels();

		// Set default voice/model only if switching *to* OpenAI and values are currently empty
		if (newEngine === 'openai') {
			if (!TTS_VOICE) TTS_VOICE = 'alloy';
			if (!TTS_MODEL) TTS_MODEL = 'tts-1';
		} // ADDED by RED
		else if (newEngine === 'customTTS_openapi') {
            // Clear the fields when switching TO custom engine
			TTS_VOICE = '';
			TTS_MODEL = '';
		}
	};

	// Runs when the component is first mounted
	onMount(async () => {
		const res = await getAudioConfig(localStorage.token);

		if (res) {
			console.log('Loaded Audio Config:', res);
			// Use nullish coalescing for safer access to potentially missing properties
			const ttsConfig = res.tts ?? {};
			const sttConfig = res.stt ?? {};

			// Load TTS settings
			TTS_OPENAI_API_BASE_URL = ttsConfig.OPENAI_API_BASE_URL ?? '';
			TTS_OPENAI_API_KEY = ttsConfig.OPENAI_API_KEY ?? '';
			// ADDED by RED: Load custom URL & Key from config
			TTS_CUSTOM_API_BASE_URL = ttsConfig.CUSTOMTTS_OPENAPI_BASE_URL ?? '';
			TTS_CUSTOM_API_KEY = ttsConfig.CUSTOMTTS_OPENAPI_KEY ?? '';

			TTS_API_KEY = ttsConfig.API_KEY ?? '';
			TTS_ENGINE = ttsConfig.ENGINE ?? '';
			TTS_MODEL = ttsConfig.MODEL ?? '';
			TTS_VOICE = ttsConfig.VOICE ?? '';
			TTS_SPLIT_ON = ttsConfig.SPLIT_ON || TTS_RESPONSE_SPLIT.PUNCTUATION;
			TTS_AZURE_SPEECH_OUTPUT_FORMAT = ttsConfig.AZURE_SPEECH_OUTPUT_FORMAT ?? '';
			TTS_AZURE_SPEECH_REGION = ttsConfig.AZURE_SPEECH_REGION ?? '';

			// Load STT settings
			STT_OPENAI_API_BASE_URL = sttConfig.OPENAI_API_BASE_URL ?? '';
			STT_OPENAI_API_KEY = sttConfig.OPENAI_API_KEY ?? '';
			STT_ENGINE = sttConfig.ENGINE ?? '';
			STT_MODEL = sttConfig.MODEL ?? '';
			STT_WHISPER_MODEL = sttConfig.WHISPER_MODEL ?? '';
			STT_AZURE_API_KEY = sttConfig.AZURE_API_KEY ?? '';
			STT_AZURE_REGION = sttConfig.AZURE_REGION ?? '';
			STT_AZURE_LOCALES = sttConfig.AZURE_LOCALES ?? '';
			STT_DEEPGRAM_API_KEY = sttConfig.DEEPGRAM_API_KEY ?? '';
		} else {
			// Set defaults if config load fails
			TTS_ENGINE = '';
			STT_ENGINE = '';
			TTS_SPLIT_ON = TTS_RESPONSE_SPLIT.PUNCTUATION;
		}

		// Fetch initial voices and models based on the loaded TTS_ENGINE
		await getVoices();
		await getModels();
	});
</script>

<!-- Main Form -->
<form
	class="flex flex-col h-full justify-between space-y-3 text-sm"
	on:submit|preventDefault={async () => {
		await updateConfigHandler();
		dispatch('save'); // Dispatch save event to parent
	}}
>
	<!-- Scrollable Content Area -->
	<div class=" space-y-3 overflow-y-scroll scrollbar-hidden h-full">
		<div class="flex flex-col gap-3">

			<!-- Speech-to-Text (STT) Settings Section (Unchanged) -->
			<div>
				<div class=" mb-1 text-sm font-medium">{$i18n.t('STT Settings')}</div>

				<!-- STT Engine Selector -->
				<div class=" py-0.5 flex w-full justify-between">
					<div class=" self-center text-xs font-medium">{$i18n.t('Speech-to-Text Engine')}</div>
					<div class="flex items-center relative">
						<select
							class="dark:bg-gray-900 cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
							bind:value={STT_ENGINE}
							aria-label={$i18n.t('Select Speech-to-Text Engine')}
						>
							<option value="">{$i18n.t('Whisper (Local)')}</option>
							<option value="openai">OpenAI</option>
							<option value="web">{$i18n.t('Web API')}</option>
							<option value="deepgram">Deepgram</option>
							<option value="azure">Azure AI Speech</option>
						</select>
					</div>
				</div>

				<!-- STT Engine Specific Settings -->
				{#if STT_ENGINE === 'openai'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<input
								class="flex-1 w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								placeholder={$i18n.t('API Base URL (e.g., https://api.openai.com/v1)')}
								bind:value={STT_OPENAI_API_BASE_URL}
								required
							/>
							<SensitiveInput placeholder={$i18n.t('API Key')} bind:value={STT_OPENAI_API_KEY} required />
						</div>
					</div>
					<hr class="border-gray-100 dark:border-gray-850 my-2" />
					<div>
						<div class=" mb-1.5 text-sm font-medium">{$i18n.t('STT Model')}</div>
						<div class="flex w-full">
							<div class="flex-1">
								<input
									list="stt-openai-model-list"
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									bind:value={STT_MODEL}
									placeholder="whisper-1"
									required
								/>
								<datalist id="stt-openai-model-list">
									<option value="whisper-1" />
								</datalist>
							</div>
						</div>
					</div>
				{:else if STT_ENGINE === 'deepgram'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<SensitiveInput placeholder={$i18n.t('API Key')} bind:value={STT_DEEPGRAM_API_KEY} required />
						</div>
					</div>
					<hr class="border-gray-100 dark:border-gray-850 my-2" />
					<div>
						<div class=" mb-1.5 text-sm font-medium">{$i18n.t('STT Model')}</div>
						<div class="flex w-full">
							<div class="flex-1">
								<input
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									bind:value={STT_MODEL}
									placeholder={$i18n.t('Select a model (optional)')}
								/>
							</div>
						</div>
						<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t('Leave model field empty to use the default model.')}
							<a
								class=" hover:underline dark:text-gray-200 text-gray-800"
								href="https://developers.deepgram.com/docs/models"
								target="_blank"
								rel="noopener noreferrer"
							>
								{$i18n.t('Click here to see available models.')}
							</a>
						</div>
					</div>
				{:else if STT_ENGINE === 'azure'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<SensitiveInput
								placeholder={$i18n.t('API Key')}
								bind:value={STT_AZURE_API_KEY}
								required
							/>
							<input
								class="flex-1 w-full rounded-lg py-2 pl-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								placeholder={$i18n.t('Azure Region')}
								bind:value={STT_AZURE_REGION}
								required
							/>
						</div>
						<hr class="border-gray-100 dark:border-gray-850 my-2" />
						<div>
							<div class=" mb-1.5 text-sm font-medium">{$i18n.t('Language Locales')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={STT_AZURE_LOCALES}
										placeholder={$i18n.t('e.g., en-US,ja-JP (leave blank for auto-detect)')}
									/>
								</div>
							</div>
						</div>
					</div>
				{:else if STT_ENGINE === ''} <!-- Whisper Local -->
					<div>
						<div class=" mb-1.5 text-sm font-medium">{$i18n.t('STT Model')}</div>
						<div class="flex w-full">
							<div class="flex-1 mr-2">
								<input
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									placeholder={$i18n.t('Set whisper model (e.g., tiny.en, base)')}
									bind:value={STT_WHISPER_MODEL}
								/>
							</div>
							<button
								type="button"
								class="px-2.5 bg-gray-50 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-gray-100 rounded-lg transition"
								on:click={sttModelUpdateHandler}
								disabled={STT_WHISPER_MODEL_LOADING}
								aria-label={$i18n.t('Download/Update Whisper Model')}
							>
								{#if STT_WHISPER_MODEL_LOADING}
									<div class="self-center animate-spin">
										<svg class=" w-4 h-4" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
											<path d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z" opacity=".25"/>
											<path d="M10.14,1.16a11,11,0,0,0-9,8.92A1.59,1.59,0,0,0,2.46,12,1.52,1.52,0,0,0,4.11,10.7a8,8,0,0,1,6.66-6.61A1.42,1.42,0,0,0,12,2.69h0A1.57,1.57,0,0,0,10.14,1.16Z" class="spinner_ajPY"/>
										</svg>
									</div>
								{:else}
									<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="w-4 h-4">
										<path d="M8.75 2.75a.75.75 0 0 0-1.5 0v5.69L5.03 6.22a.75.75 0 0 0-1.06 1.06l3.5 3.5a.75.75 0 0 0 1.06 0l3.5-3.5a.75.75 0 0 0-1.06-1.06L8.75 8.44V2.75Z" />
										<path d="M3.5 9.75a.75.75 0 0 0-1.5 0v1.5A2.75 2.75 0 0 0 4.75 14h6.5A2.75 2.75 0 0 0 14 11.25v-1.5a.75.75 0 0 0-1.5 0v1.5c0 .69-.56 1.25-1.25 1.25h-6.5c-.69 0-1.25-.56-1.25-1.25v-1.5Z" />
									</svg>
								{/if}
							</button>
						</div>
						<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t(`Open WebUI uses faster-whisper internally.`)}
							<a
								class=" hover:underline dark:text-gray-200 text-gray-800"
								href="https://github.com/SYSTRAN/faster-whisper#model-conversion"
								target="_blank"
								rel="noopener noreferrer"
							>
								{$i18n.t(`Click here to learn more about faster-whisper and see the available models.`)}
							</a>
						</div>
					</div>
				{/if} <!-- End STT Engine Conditional Blocks -->
			</div> <!-- End STT Section -->

			<hr class="border-gray-100 dark:border-gray-850" />

			<!-- Text-to-Speech (TTS) Settings Section -->
			<div>
				<div class=" mb-1 text-sm font-medium">{$i18n.t('TTS Settings')}</div>

				<!-- TTS Engine Selector -->
				<div class=" py-0.5 flex w-full justify-between">
					<div class=" self-center text-xs font-medium">{$i18n.t('Text-to-Speech Engine')}</div>
					<div class="flex items-center relative">
						<select
							class=" dark:bg-gray-900 w-fit pr-8 cursor-pointer rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
							bind:value={TTS_ENGINE}
							aria-label={$i18n.t('Select Text-to-Speech Engine')}
							on:change={handleTtsEngineChange}
						>
							<option value="">{$i18n.t('Web API')}</option>
							<option value="transformers">{$i18n.t('Transformers')} ({$i18n.t('Local')})</option>
							<option value="openai">{$i18n.t('OpenAI')}</option>
							<option value="elevenlabs">{$i18n.t('ElevenLabs')}</option>
							<option value="azure">{$i18n.t('Azure AI Speech')}</option>
							<!-- ADDED by RED: Custom TTS option in dropdown -->
							<option value="customTTS_openapi">{$i18n.t('Custom TTS')}</option>
						</select>
					</div>
				</div>

				<!-- TTS Engine Specific API Key / URL / Region Inputs -->
				{#if TTS_ENGINE === 'openai'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<input
								class="flex-1 w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								placeholder={$i18n.t('API Base URL (e.g., https://api.openai.com/v1)')}
								bind:value={TTS_OPENAI_API_BASE_URL}
								required
							/>
							<SensitiveInput placeholder={$i18n.t('API Key')} bind:value={TTS_OPENAI_API_KEY} required />
						</div>
					</div>
				{:else if TTS_ENGINE === 'elevenlabs'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<SensitiveInput placeholder={$i18n.t('API Key (xi-api-key)')} bind:value={TTS_API_KEY} required />
						</div>
					</div>
				{:else if TTS_ENGINE === 'azure'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<SensitiveInput placeholder={$i18n.t('API Key')} bind:value={TTS_API_KEY} required />
							<input
								class="flex-1 w-full rounded-lg py-2 pl-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								placeholder={$i18n.t('Azure Region')}
								bind:value={TTS_AZURE_SPEECH_REGION}
								required
							/>
						</div>
					</div>
				<!-- ADDED by RED: Section for Custom TTS Engine Configuration -->
				{:else if TTS_ENGINE === 'customTTS_openapi'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<input
								class="flex-1 w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								placeholder={$i18n.t('API Base URL (OpenAI format)')}
								bind:value={TTS_CUSTOM_API_BASE_URL}
								required
							/>
							<SensitiveInput placeholder={$i18n.t('API Key (Optional)')} bind:value={TTS_CUSTOM_API_KEY} />
						</div>
						<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t('Your custom endpoint should mimic the OpenAI /v1/audio/speech endpoint.')}
							<a
								class=" hover:underline dark:text-gray-200 text-gray-800"
								href="https://platform.openai.com/docs/api-reference/audio/createSpeech"
								target="_blank"
								rel="noopener noreferrer"
							>
								{$i18n.t('OpenAI API Reference')}
							</a>
						</div>
					</div>
				{/if} <!-- End of engine-specific API key/URL/Region inputs -->

				<!-- Separator (only show if an engine requiring settings above is selected) -->
				{#if TTS_ENGINE !== ''}
					<hr class="border-gray-100 dark:border-gray-850 my-2" />
				{/if}

				<!-- TTS Voice and Model/Format Selection -->
				{#if TTS_ENGINE === ''} <!-- Web API Voice Selection -->
					<div>
						<div class=" mb-1.5 text-sm font-medium">{$i18n.t('TTS Voice')}</div>
						<div class="flex w-full">
							<div class="flex-1">
								<select
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden cursor-pointer"
									bind:value={TTS_VOICE}
									aria-label={$i18n.t('Select Web API TTS Voice')}
								>
									<option value="" selected={!TTS_VOICE}>{$i18n.t('Default Browser Voice')}</option>
									{#if Array.isArray(voices)}
										{#each voices as voice}
											{#if 'voiceURI' in voice} <!-- Check for browser voice properties -->
												<option value={voice.name} selected={TTS_VOICE === voice.name}>
													{voice.name} ({voice.lang})
												</option>
											{/if}
										{/each}
									{/if}
								</select>
							</div>
						</div>
					</div>
				{:else if TTS_ENGINE === 'transformers'}
					<div>
						<div class=" mb-1.5 text-sm font-medium">{$i18n.t('TTS Model')}</div>
						<div class="flex w-full">
							<div class="flex-1">
								<input
									list="model-list-transformers"
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									bind:value={TTS_MODEL}
									placeholder="CMU ARCTIC speaker embedding name"
								/>
								<!-- FIXED by RED: Moved comment outside the input tag -->
								<!-- Changed list id slightly -->
								<datalist id="model-list-transformers">
									<!-- No options typically listed here unless specific common ones are known -->
								</datalist>
							</div>
						</div>
						<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t(`Open WebUI uses SpeechT5 and CMU Arctic speaker embeddings.`)}
							To learn more about SpeechT5,
							<a
								class=" hover:underline dark:text-gray-200 text-gray-800"
								href="https://github.com/microsoft/SpeechT5"
								target="_blank"
								rel="noopener noreferrer"
							>
								{$i18n.t(`click here`, { name: 'SpeechT5' })}.
							</a>
							To see the available CMU Arctic speaker embeddings,
							<a
								class=" hover:underline dark:text-gray-200 text-gray-800"
								href="https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors"
								target="_blank"
								rel="noopener noreferrer"
							>
								{$i18n.t(`click here`)}.
							</a>
						</div>
					</div>
				{:else if TTS_ENGINE === 'openai' || TTS_ENGINE === 'elevenlabs'}
					<!-- Voice/Model for OpenAI, ElevenLabs -->
					<div class=" flex gap-2">
						<div class="w-full">
							<div class=" mb-1.5 text-sm font-medium">{$i18n.t('TTS Voice')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										list={`tts-${TTS_ENGINE}-voice-list`}
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_VOICE}
										placeholder={$i18n.t('Select or enter Voice ID')}
										required
									/>
									<datalist id={`tts-${TTS_ENGINE}-voice-list`}>
										{#if Array.isArray(voices)}
											{#each voices as voice}
												{#if 'id' in voice} <!-- Check for backend voice properties -->
													<option value={voice.id}>{voice.name}</option>
												{/if}
											{/each}
										{/if}
										<!-- Static defaults only for OpenAI -->
										{#if TTS_ENGINE === 'openai'}
											<option value="alloy">alloy</option>
											<option value="echo">echo</option>
											<option value="fable">fable</option>
											<option value="onyx">onyx</option>
											<option value="nova">nova</option>
											<option value="shimmer">shimmer</option>
										{/if}
									</datalist>
								</div>
							</div>
						</div>
						<div class="w-full">
							<div class=" mb-1.5 text-sm font-medium">{$i18n.t('TTS Model')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										list={`tts-${TTS_ENGINE}-model-list`}
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_MODEL}
										placeholder={$i18n.t('Select or enter Model ID')}
										required
									/>
									<datalist id={`tts-${TTS_ENGINE}-model-list`}>
										{#if Array.isArray(models)}
											{#each models as model}
												<option value={model.id} />
											{/each}
										{/if}
										<!-- Static defaults only for OpenAI -->
										{#if TTS_ENGINE === 'openai'}
											<option value="tts-1">tts-1</option>
											<option value="tts-1-hd">tts-1-hd</option>
										{/if}
									</datalist>
								</div>
							</div>
						</div>
					</div>
				<!-- ADDED by RED: Custom TTS now uses dynamic lists from API -->
				{:else if TTS_ENGINE === 'customTTS_openapi'}
					<!-- Voice/Model for Custom TTS -->
					<div class=" flex gap-2">
						<div class="w-full">
							<div class=" mb-1.5 text-sm font-medium">{$i18n.t('TTS Voice')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										list={`tts-${TTS_ENGINE}-voice-list`}
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_VOICE}
										placeholder={$i18n.t('Select or enter Voice ID')}
										required
									/>
									<datalist id={`tts-${TTS_ENGINE}-voice-list`}>
										{#if Array.isArray(voices)}
											{#each voices as voice}
												{#if 'id' in voice} <!-- Check for backend voice properties -->
													<option value={voice.id}>{voice.name}</option>
												{/if}
											{/each}
										{/if}
										<!-- No static defaults here, list comes from API -->
									</datalist>
								</div>
							</div>
						</div>
						<div class="w-full">
							<div class=" mb-1.5 text-sm font-medium">{$i18n.t('TTS Model')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										list={`tts-${TTS_ENGINE}-model-list`}
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_MODEL}
										placeholder={$i18n.t('Select or enter Model ID')}
										required
									/>
									<datalist id={`tts-${TTS_ENGINE}-model-list`}>
										{#if Array.isArray(models)}
											{#each models as model}
												<option value={model.id} />
											{/each}
										{/if}
										<!-- No static defaults here, list comes from API -->
									</datalist>
								</div>
							</div>
						</div>
					</div>
				{:else if TTS_ENGINE === 'azure'}
					<!-- Voice/Output Format for Azure -->
					<div class=" flex gap-2">
						<div class="w-full">
							<div class=" mb-1.5 text-sm font-medium">{$i18n.t('TTS Voice')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										list="tts-azure-voice-list"
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_VOICE}
										placeholder={$i18n.t('Select or enter voice name')}
										required
									/>
									<datalist id="tts-azure-voice-list">
										{#if Array.isArray(voices)}
											{#each voices as voice}
												{#if 'id' in voice} <!-- Check for backend voice -->
													<option value={voice.id}>{voice.name}</option>
												{/if}
											{/each}
										{/if}
									</datalist>
								</div>
							</div>
						</div>
						<div class="w-full">
							<div class=" mb-1.5 text-sm font-medium">
								{$i18n.t('Output format')}
								<a
									class="hover:underline dark:text-gray-200 text-gray-800 text-xs ml-1"
									href="https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech?tabs=streaming#audio-outputs"
									target="_blank"
									rel="noopener noreferrer"
								>
									({$i18n.t('Available list')})
								</a>
							</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_AZURE_SPEECH_OUTPUT_FORMAT}
										placeholder={$i18n.t('e.g., audio-24khz-160kbitrate-mono-mp3')}
										required
									/>
								</div>
							</div>
						</div>
					</div>
				{/if} <!-- End of voice/model/format selection blocks -->

				<!-- Common Response Splitting Setting -->
				<hr class="border-gray-100 dark:border-gray-850 my-2" />
				<div class="pt-0.5 flex w-full justify-between">
					<div class="self-center text-xs font-medium">{$i18n.t('Response splitting')}</div>
					<div class="flex items-center relative">
						<select
							class="dark:bg-gray-900 w-fit pr-8 cursor-pointer rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
							aria-label={$i18n.t('Select how to split message text for TTS requests')}
							bind:value={TTS_SPLIT_ON}
						>
							{#each Object.values(TTS_RESPONSE_SPLIT) as split}
								<option value={split}>
									{$i18n.t(split.charAt(0).toUpperCase() + split.slice(1))}
								</option>
							{/each}
						</select>
					</div>
				</div>
				<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
					{$i18n.t(
						"Control how message text is split for TTS requests. 'Punctuation' splits into sentences, 'New line' splits by line breaks, and 'None' keeps the message as a single string."
					)}
				</div>
			</div> <!-- End TTS Section -->

		</div> <!-- End Main Content Flex Col -->
	</div> <!-- End Scrollable Area -->

	<!-- Form Actions -->
	<div class="flex justify-end text-sm font-medium">
		<button
			class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full"
			type="submit"
		>
			{$i18n.t('Save')}
		</button>
	</div>
</form>