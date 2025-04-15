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
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	import { TTS_RESPONSE_SPLIT } from '$lib/types';

	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';

	const i18n = getContext<Writable<i18nType>>('i18n');

	export let saveHandler: () => void;

	// Audio State Variables
	let TTS_OPENAI_API_BASE_URL = '';
	let TTS_OPENAI_API_KEY = '';
	let TTS_CUSTOM_API_BASE_URL = '';
	let TTS_CUSTOM_API_KEY = '';
	let TTS_API_KEY = ''; // Used by ElevenLabs, Azure
	let TTS_ENGINE = '';
	let TTS_MODEL = '';
	let TTS_VOICE = '';
	let TTS_SPLIT_ON: TTS_RESPONSE_SPLIT = TTS_RESPONSE_SPLIT.PUNCTUATION;
	let TTS_AZURE_SPEECH_REGION = '';
	let TTS_AZURE_SPEECH_OUTPUT_FORMAT = '';

	// STT State Variables
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

	// Dynamic lists state
	// eslint-disable-next-line no-undef
	let voices: (SpeechSynthesisVoice | { id: string; name: string; [key: string]: any })[] = [];
	let models: Awaited<ReturnType<typeof _getModels>>['models'] = [];

	// Fetches models using saved config (no verify params)
	const getModels = async () => {
		if (TTS_ENGINE === '' || TTS_ENGINE === 'transformers') {
			models = [];
			return;
		}
		try {
			const res = await _getModels(localStorage.token);
			if (res) {
				console.log('Models response:', res);
				models = res.models || [];
			} else {
				models = [];
			}
		} catch (e) {
			toast.error(`Error fetching models: ${e}`);
			models = [];
		}
	};

	// Fetches voices using saved config (no verify params)
	const getVoices = async () => {
		if (TTS_ENGINE === '') {
			voices = [];
			const getVoicesLoop = setInterval(() => {
				const browserVoices = speechSynthesis.getVoices();
				if (browserVoices.length > 0) {
					clearInterval(getVoicesLoop);
					voices = browserVoices.sort((a, b) =>
						a.name.localeCompare(b.name, $i18n.resolvedLanguage)
					);
				}
			}, 100);
		} else if (TTS_ENGINE === 'transformers') {
			voices = [];
		} else {
			voices = [];
			try {
				const res = await _getVoices(localStorage.token);
				if (res) {
					console.log('Voices response:', res);
					voices = res.voices || [];
					if (Array.isArray(voices)) {
						voices.sort((a, b) => a.name.localeCompare(b.name, $i18n.resolvedLanguage));
					}
				} else {
					voices = [];
				}
			} catch (e) {
				toast.error(`Error fetching voices: ${e}`);
				voices = [];
			}
		}
	};

	// Saves the entire configuration to the backend
	const updateConfigHandler = async () => {
		console.log('Updating config with TTS Engine:', TTS_ENGINE);
		const payload = {
			tts: {
				OPENAI_API_BASE_URL: TTS_OPENAI_API_BASE_URL,
				OPENAI_API_KEY: TTS_OPENAI_API_KEY,
				CUSTOMTTS_OPENAPI_BASE_URL: TTS_CUSTOM_API_BASE_URL,
				CUSTOMTTS_OPENAPI_KEY: TTS_CUSTOM_API_KEY,
				API_KEY: TTS_API_KEY,
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
		};
		console.log("Sending Payload to Backend:", payload);

		try {
			const res = await updateAudioConfig(localStorage.token, payload);
			if (res) {
				toast.success($i18n.t('Configuration saved successfully!'));
				saveHandler();
				config.set(await getBackendConfig());
				await getVoices(); // Re-fetch based on potentially new saved state
				await getModels(); // Re-fetch based on potentially new saved state
			} else {
				toast.error($i18n.t('Failed to save configuration. Please check backend logs.'));
			}
		} catch (error) {
			console.error("Error saving config:", error);
			toast.error(`${$i18n.t('Error saving configuration:')} ${error.detail ?? error}`);
		}
	};

	// Handler for Custom TTS Key blur / Verify button click
	const handleCustomTTSConfigChange = async () => {
		if (TTS_ENGINE === 'customTTS_openapi' && TTS_CUSTOM_API_BASE_URL) {
			console.log('Fetching custom TTS models/voices triggered...');
			const fetchToastId = toast.loading($i18n.t('Verifying connection & fetching lists...'));

			let modelsLoaded = false;
			let voicesLoaded = false;

			try {
				const modelRes = await _getModels(
					localStorage.token,
					TTS_CUSTOM_API_BASE_URL,
					TTS_CUSTOM_API_KEY
				);
				if (modelRes && Array.isArray(modelRes.models)) {
					models = modelRes.models;
					modelsLoaded = true;
				} else {
					toast.error($i18n.t('Failed to load models: Unexpected response format.'), { id: fetchToastId });
					models = [];
				}
			} catch (error) {
     			console.error('>>> DEBUG: Raw error caught in handler:', error); // <-- ADD RAW LOG
     			console.error('Verification Error:', error); // Keep original log
     			toast.error(typeof error === 'string' ? error : $i18n.t('Verification failed. Check URL/Key and backend logs.'), {
         			id: fetchToastId
     			});
     			models = [];
     			voices = [];
}

			try {
				const voiceRes = await _getVoices(
					localStorage.token,
					TTS_CUSTOM_API_BASE_URL,
					TTS_CUSTOM_API_KEY
				);
				if (voiceRes && Array.isArray(voiceRes.voices)) {
					voices = voiceRes.voices.sort((a, b) => a.name.localeCompare(b.name, $i18n.resolvedLanguage));
					voicesLoaded = true;
				} else {
					toast.error($i18n.t('Failed to load voices: Unexpected response format.'), { id: fetchToastId });
					voices = [];
				}
			} catch (error) {
				console.error('Verification/Fetching Voices Error:', error);
				toast.error(typeof error === 'string' ? error : $i18n.t('Failed to fetch voices. Check URL/Key and logs.'), {
					id: fetchToastId
				});
				voices = [];
			}

			if (modelsLoaded && voicesLoaded) {
				toast.success($i18n.t('Connection verified. Models and voices loaded.'), { id: fetchToastId, duration: 2500 });
			} else if (modelsLoaded && !voicesLoaded) {
				toast.dismiss(fetchToastId);
			} else if (!modelsLoaded && voicesLoaded) {
                toast.dismiss(fetchToastId);
            } else {
                toast.dismiss(fetchToastId);
			}

		} else if (TTS_ENGINE === 'customTTS_openapi' && !TTS_CUSTOM_API_BASE_URL) {
			toast.info($i18n.t('Please enter the Base URL first.'));
		}
	};

	// Handler for the local Whisper model update/download button
	const sttModelUpdateHandler = async () => {
		STT_WHISPER_MODEL_LOADING = true;
		await updateConfigHandler();
		STT_WHISPER_MODEL_LOADING = false;
	};

	// Handler for changes to the TTS engine dropdown
	const handleTtsEngineChange = async (event: Event & { currentTarget: HTMLSelectElement }) => {
		const newEngine = event.currentTarget.value;
		const oldEngine = TTS_ENGINE;
		TTS_ENGINE = newEngine;

		await getVoices(); // Uses standard backend state
		await getModels(); // Uses standard backend state

		if (newEngine === 'openai') {
			if (!TTS_VOICE) TTS_VOICE = 'alloy';
			if (!TTS_MODEL) TTS_MODEL = 'tts-1';
		} else if (newEngine === 'customTTS_openapi') {
			if (oldEngine !== 'customTTS_openapi') {
				TTS_VOICE = '';
				TTS_MODEL = '';
				if (TTS_CUSTOM_API_BASE_URL) {
					// Automatically trigger verify/fetch if URL exists when switching TO custom
					await handleCustomTTSConfigChange();
				}
			}
		}
	};

	// Runs when the component is first mounted
	onMount(async () => {
		const res = await getAudioConfig(localStorage.token);

		if (res) {
			console.log('Loaded Audio Config:', res);
			const ttsConfig = res.tts ?? {};
			const sttConfig = res.stt ?? {};

			// Load TTS settings
			TTS_OPENAI_API_BASE_URL = ttsConfig.OPENAI_API_BASE_URL ?? '';
			TTS_OPENAI_API_KEY = ttsConfig.OPENAI_API_KEY ?? '';
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
			TTS_ENGINE = '';
			STT_ENGINE = '';
			TTS_SPLIT_ON = TTS_RESPONSE_SPLIT.PUNCTUATION;
		}

		// Fetch initial voices and models based on the loaded TTS_ENGINE
		await getVoices();
		await getModels();

		// Apply initial OpenAI defaults ONLY if engine is OpenAI and fields are empty
		if (TTS_ENGINE === 'openai') {
			if (!TTS_VOICE) TTS_VOICE = 'alloy';
			if (!TTS_MODEL) TTS_MODEL = 'tts-1';
		}
	});
 </script>

 <!-- Main Form -->
 <form
	class="flex flex-col h-full justify-between space-y-3 text-sm"
	on:submit|preventDefault={async () => {
		await updateConfigHandler();
	}}
 >
	 <!-- Scrollable Content Area -->
	 <div class=" space-y-3 overflow-y-scroll scrollbar-hidden h-full">
		 <div class="flex flex-col gap-3">
 
			<!-- Speech-to-Text (STT) Settings Section -->
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
				 {/if}
			 </div>
 
			 <hr class="border-gray-100 dark:border-gray-850" />
 
			 <!-- TTS Section -->
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
							 <option value="customTTS_openapi">{$i18n.t('Custom (OpenAPI)')}</option>
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
 
				 {:else if TTS_ENGINE === 'customTTS_openapi'}
					 <div>
						 <div class="mt-1 flex flex-wrap gap-2 items-end">
							 <div class="flex-1 min-w-[200px]">
								  <label for="custom-tts-url" class="block mb-1 text-xs text-gray-500">{$i18n.t('API Base URL (OpenAI format)')}</label>
								 <input
									 id="custom-tts-url"
									 class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									 placeholder="http://localhost:8880/v1"
									 bind:value={TTS_CUSTOM_API_BASE_URL}
									 required
								 />
							 </div>
 
							 <div class="flex-1 min-w-[200px] flex items-end gap-1">
								 <div class="flex-1">
									 <label for="custom-tts-key" class="block mb-1 text-xs text-gray-500">{$i18n.t('API Key (Optional)')}</label>
									 <SensitiveInput
										 inputId="custom-tts-key"
										 placeholder={$i18n.t('API Key')}
										 bind:value={TTS_CUSTOM_API_KEY}
										 on:blur={handleCustomTTSConfigChange}
									  />
								 </div>
 
								 <div class="shrink-0">
									<Tooltip content={$i18n.t('Verify & Fetch Lists')} className="-mb-[1px]">
									   <button
										  on:click={handleCustomTTSConfigChange}
										  type="button"
										  class="p-2 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded-lg transition"
										  aria-label={$i18n.t('Verify & Fetch Lists')}
									   >
										  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
											 <path fill-rule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z" clip-rule="evenodd" />
										  </svg>
									   </button>
									</Tooltip>
								 </div>
							 </div>
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
				 {/if}
 
				 {#if TTS_ENGINE !== ''}
					 <hr class="border-gray-100 dark:border-gray-850 my-2" />
				 {/if}
 
				 {#if TTS_ENGINE === ''}
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
											 {#if 'voiceURI' in voice}
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
								 <datalist id="model-list-transformers"></datalist>
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
				 {:else if TTS_ENGINE === 'openai' || TTS_ENGINE === 'elevenlabs' || TTS_ENGINE === 'customTTS_openapi'}
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
										 required={TTS_ENGINE !== 'customTTS_openapi'}
									 />
									 <datalist id={`tts-${TTS_ENGINE}-voice-list`}>
										 {#if Array.isArray(voices)}
											 {#each voices as voice}
												 {#if 'id' in voice}
													 <option value={voice.id}>{voice.name}</option>
												 {/if}
											 {/each}
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
										 required={TTS_ENGINE !== 'customTTS_openapi'}
									 />
									 <datalist id={`tts-${TTS_ENGINE}-model-list`}>
										 {#if Array.isArray(models)}
											 {#each models as model}
												 <option value={model.id}>{model.name ?? model.id}</option>
											 {/each}
										 {/if}
									 </datalist>
								 </div>
							 </div>
						 </div>
					 </div>
				 {:else if TTS_ENGINE === 'azure'}
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
												 {#if 'id' in voice}
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
				 {/if}
 
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
			 </div>
 
		 </div>
	 </div>
 
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