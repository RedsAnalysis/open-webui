// /static/vad-processor.js

class VADProcessor extends AudioWorkletProcessor {
    constructor(options) {
      super(options);
      // You could receive options here using options.processorOptions
      console.log('VADProcessor constructor called');
    }
  
    process(inputs, outputs, parameters) {
      // inputs[0] refers to the first input buffer
      // inputs[0][0] refers to the first channel of the first input buffer (Float32Array)
      const inputChannelData = inputs[0]?.[0];
  
      // Check if there's valid input data
      if (inputChannelData && inputChannelData.length > 0) {
        // Send the raw audio data (Float32Array) back to the main thread
        // Note: Transferring the buffer might be slightly more efficient
        // if the main thread doesn't need the original buffer anymore,
        // but postMessage is simpler for now.
        this.port.postMessage(inputChannelData);
      }
  
      // Return true to keep the processor alive
      return true;
    }
  }
  
  try {
      registerProcessor('vad-processor', VADProcessor);
  } catch (e) {
      console.error("Error registering VADProcessor:", e);
  }