// src/lib/utils/vad.ts
import type * as ort from 'onnxruntime-web';

// Interfaces for ONNX model input/output names.
// IMPORTANT: These names ('input', 'sr', 'h', 'c', 'output', 'hn', 'cn')
// MUST match the actual input/output names of your specific
// silero_vad_16k_op15.onnx model. Use a tool like Netron (https://netron.app/)
// to view the model and confirm these names if you encounter errors.
interface VadInput {
	input: ort.Tensor;
	sr: ort.Tensor;
	h: ort.Tensor;
	c: ort.Tensor;
}

interface VadOutput {
	output: ort.Tensor; // Speech probability score
	hn: ort.Tensor;     // Next hidden state
	cn: ort.Tensor;     // Next cell state
}

export class vad {
	private session: ort.InferenceSession;
	private ort: typeof ort; // Reference to the imported onnxruntime-web library
	private _h: ort.Tensor;  // Hidden state tensor
	private _c: ort.Tensor;  // Cell state tensor
	private srTensor: ort.Tensor; // Sample rate tensor

	// Model-specific parameters (adjust if necessary based on model inspection)
	private readonly sampleRate = 16000;
	// This frame/chunk size should ideally match the expected input dimension
	// of the 'input' tensor in your ONNX model. Common Silero sizes: 512, 1024, 1536.
	// We used 512 in the AudioWorkletNode setup, let's keep it consistent here.
	private readonly frameSize = 512;

	constructor(session: ort.InferenceSession, ortRef: typeof ort) {
		this.session = session;
		this.ort = ortRef;
		// Create the sample rate tensor (int64). The VAD model might ignore this
		// if the sample rate is baked in, but it's often required by the ONNX graph structure.
		this.srTensor = new ort.Tensor('int64', [BigInt(this.sampleRate)]);
		this.reset_state(); // Initialize hidden states
		console.log('VAD class instantiated.');
	}

	/**
	 * Resets the hidden and cell states of the VAD model.
	 * Should be called before starting a new detection sequence or after an error.
	 */
	reset_state() {
		console.log('Resetting VAD hidden states.');
		// Dimensions depend on the specific VAD model architecture (LSTM/GRU layers, hidden size).
		// Common Silero VAD uses LSTM layers. Check your model via Netron if unsure.
		// D = 1 for non-bidirectional LSTM/GRU
		// num_layers = Number of LSTM layers (e.g., 1 or 2)
		// N = Batch size (must be 1 for real-time frame-by-frame processing)
		// H = Hidden size (e.g., 64 or 128 for Silero)
		const D_NUM_LAYERS = 1; // Example: Assume 1 layer
		const BATCH_SIZE = 1;
		const HIDDEN_SIZE = 64; // Example: Assume hidden size 64

		// Shape: [D*num_layers, N, H] = [1, 1, 64]
		const stateShape: ReadonlyArray<number> = [D_NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE];
		const stateSize = D_NUM_LAYERS * BATCH_SIZE * HIDDEN_SIZE;

		this._h = new this.ort.Tensor('float32', new Float32Array(stateSize).fill(0), stateShape);
		this._c = new this.ort.Tensor('float32', new Float32Array(stateSize).fill(0), stateShape);
	}

	/**
	 * Processes a single audio frame (chunk) and returns the speech probability.
	 * @param audioFrame A Float32Array containing the audio data for the frame.
	 * @returns A number between 0 and 1 representing the speech probability.
	 */
	process(audioFrame: Float32Array): number {
		// --- Input Validation & Preparation ---
		if (!audioFrame) {
			console.warn('VAD process called with null or undefined audioFrame.');
			return 0;
		}

		// Ensure the input frame matches the expected size (this might require buffering upstream).
		// For simplicity here, we pad/truncate if the AudioWorkletNode doesn't provide the exact size.
		let frameToProcess: Float32Array;
		if (audioFrame.length !== this.frameSize) {
			// console.warn(`VAD input frame size mismatch: expected ${this.frameSize}, got ${audioFrame.length}. Padding/truncating.`);
			frameToProcess = new Float32Array(this.frameSize).fill(0);
			const len = Math.min(audioFrame.length, this.frameSize);
			frameToProcess.set(audioFrame.slice(0, len));
		} else {
			frameToProcess = audioFrame;
		}

		// Create the main input tensor: Shape [batch_size, sequence_length] = [1, frameSize]
		const inputTensor = new this.ort.Tensor('float32', frameToProcess, [1, this.frameSize]);

		// --- Prepare Feeds Dictionary ---
		// Keys MUST match the actual input names of your ONNX model.
		const feeds: VadInput = {
			input: inputTensor,
			sr: this.srTensor,
			h: this._h,
			c: this._c
		};

		// --- Run Inference ---
		try {
			// Run inference (synchronous with WASM backend)
			const results = this.session.run(feeds) as VadOutput;

			// --- Update Hidden States ---
			// Output names ('hn', 'cn') MUST match your model.
			this._h = results.hn;
			this._c = results.cn;

			// --- Extract Probability ---
			// Output name ('output') MUST match your model.
			const outputTensor = results.output;
			// Output data is likely a Float32Array. Access the first element.
			const probability = (outputTensor.data as Float32Array)[0];

			// Clamp probability between 0 and 1 just in case.
			return Math.max(0, Math.min(1, probability));

		} catch (e) {
			console.error('Error running VAD inference:', e);
			this.reset_state(); // Reset state on error
			return 0; // Return 0 probability on error
		}
	}
}