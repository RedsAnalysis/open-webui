import { AUDIO_API_BASE_URL } from '$lib/constants';

export const getAudioConfig = async (token: string) => {
	let error = null;

	const res = await fetch(`${AUDIO_API_BASE_URL}/config`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.log(err);
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

type OpenAIConfigForm = {
	url: string;
	key: string;
	model: string;
	speaker: string;
};

export const updateAudioConfig = async (token: string, payload: OpenAIConfigForm) => {
	let error = null;

	const res = await fetch(`${AUDIO_API_BASE_URL}/config/update`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			...payload
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.log(err);
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const transcribeAudio = async (token: string, file: File) => {
	const data = new FormData();
	data.append('file', file);

	let error = null;
	const res = await fetch(`${AUDIO_API_BASE_URL}/transcriptions`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			authorization: `Bearer ${token}`
		},
		body: data
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.log(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const synthesizeOpenAISpeech = async (
	token: string = '',
	speaker: string = 'alloy',
	text: string = '',
	model?: string
) => {
	let error = null;

	const res = await fetch(`${AUDIO_API_BASE_URL}/speech`, {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${token}`,
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			input: text,
			voice: speaker,
			...(model && { model })
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res;
		})
		.catch((err) => {
			error = err.detail;
			console.log(err);

			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

interface AvailableModelsResponse {
	models: { name: string; id: string }[] | { id: string }[];
}

// Define the expected response structure more explicitly (optional but good practice)
interface ModelData {
	id: string;
	name?: string; // Name is optional now
	[key: string]: any; // Allow other properties
}

interface AvailableModelsResponse {
	models: ModelData[];
}


// Modified function
export const getModels = async (
	token: string = '',
	// ADDED verify parameters (optional)
	verifyUrl?: string,
	verifyKey?: string
): Promise<AvailableModelsResponse> => { // Ensure return type matches interface
	let error: string | null = null; // Explicitly type error as string or null

	// Base URL from constant
	let url = `${AUDIO_API_BASE_URL}/models`;
	const fetchOptions: RequestInit = { // Define fetch options
		method: 'GET',
		headers: {
			Accept: 'application/json', // Prefer Accept over Content-Type for GET
			Authorization: `Bearer ${token}`
		}
	}

	// Conditionally add verify parameters as query string
	if (verifyUrl) {
		const params = new URLSearchParams();
		params.append('verify_url', verifyUrl);
		if (verifyKey) { // Only add key if provided and not empty
			params.append('verify_key', verifyKey);
		}
		url += `?${params.toString()}`;
		console.log(`Verification call to: ${url}`);
	}

	console.log(`Fetching models from: ${url}`); // Log the final URL

	const res = await fetch(url, fetchOptions) // Use the potentially modified URL and options
		.then(async (res) => {
			// If verification failed, backend raises HTTPException which results in !res.ok
			if (!res.ok) {
				// Try to parse error detail from backend
				try {
					const errData = await res.json();
					// Use detail if available, otherwise use status text
					throw new Error(errData.detail ?? `${res.status} ${res.statusText}`);
				} catch (parseError) {
					// Fallback if parsing error fails
					throw new Error(`${res.status} ${res.statusText}`);
				}
			}
			return res.json();
		})
		.catch((err) => {
			// err should now be an Error object
			if (verifyUrl) {
				// Prepend context for verification errors
				error = `Verification failed: ${err.message}`;
			} else {
				error = `Error fetching models: ${err.message}`;
			}
			console.error(error); // Log the formatted error
			return null; // Return null to indicate failure
		});

	if (error) {
		throw error; // Re-throw the formatted error string
	}

	// Ensure the response structure matches, provide default if needed
	if (res && Array.isArray(res.models)) {
		return res;
	} else {
		console.warn('Received unexpected format for models, returning empty array.');
		return { models: [] }; // Return valid structure with empty array
	}
};

interface VoiceData {
	id: string;
	name: string; // Name is expected from backend formatting now
	[key: string]: any; // Allow other properties
}
interface AvailableVoicesResponse {
	voices: VoiceData[];
}


// Modified function
export const getVoices = async (
	token: string = '',
	// ADDED verify parameters (optional)
	verifyUrl?: string,
	verifyKey?: string
): Promise<AvailableVoicesResponse> => { // Ensure return type matches interface
	let error: string | null = null; // Explicitly type error as string or null

	// Base URL from constant
	let url = `${AUDIO_API_BASE_URL}/voices`;
	const fetchOptions: RequestInit = { // Define fetch options
		method: 'GET',
		headers: {
			Accept: 'application/json', // Prefer Accept over Content-Type for GET
			Authorization: `Bearer ${token}`
		}
	}

	// Conditionally add verify parameters as query string
	if (verifyUrl) {
		const params = new URLSearchParams();
		params.append('verify_url', verifyUrl);
		if (verifyKey) { // Only add key if provided and not empty
			params.append('verify_key', verifyKey);
		}
		url += `?${params.toString()}`;
		console.log(`Verification call to: ${url}`);
	}

	console.log(`Fetching voices from: ${url}`); // Log the final URL

	const res = await fetch(url, fetchOptions) // Use the potentially modified URL and options
		.then(async (res) => {
			// If verification failed, backend raises HTTPException which results in !res.ok
			if (!res.ok) {
				try {
					const errData = await res.json();
					throw new Error(errData.detail ?? `${res.status} ${res.statusText}`);
				} catch (parseError) {
					throw new Error(`${res.status} ${res.statusText}`);
				}
			}
			return res.json();
		})
		.catch((err) => {
			if (verifyUrl) {
				error = `Verification failed: ${err.message}`;
			} else {
				error = `Error fetching voices: ${err.message}`;
			}
			console.error(error);
			return null; // Return null to indicate failure
		});

	if (error) {
		throw error; // Re-throw the formatted error string
	}

	// Ensure the response structure matches, provide default if needed
	if (res && Array.isArray(res.voices)) {
		return res;
	} else {
		console.warn('Received unexpected format for voices, returning empty array.');
		return { voices: [] }; // Return valid structure with empty array
	}
};