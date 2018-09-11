//Programmer: Chris Tralie

/** A canvas for drawing a self-similarity matrix synchronized with audio */

/**
 * Create a structure canvas
 * @param {object} audio_obj - A dictionary of audio parameters, including 
 * 	a handle to the audio widget and a time interval between adjacent rows
 *  of the SSM, and the dimension "dim" of the SSM
 */
function ForceCanvas(audio_obj) {


	this.audio_obj = audio_obj;

	/**
	 * A function use to update the images on the canvas and to
	 * resize things accordingly
	 * @param {object} params : A dictionary of parameters returned
	 * from the Python program as a JSON file
	 */
	this.updateParams = function(params) {
		/** TODO: Finish this */
		requestAnimationFrame(this.repaint.bind(this));
	};

	/**
	 * A function which toggles all of the visible elements to show
	 */
	this.show = function() {
		console.log("Showing force display");
	};

	/**
	 * A function which toggles all of the visible elements to hide
	 */
	this.hide = function() {
		console.log("Hiding force display");
	};


	/**
	 * A fuction which renders the SSM and laplacian eigenvectors
	 * with lines superimposed to show where the audio is
	 */
	this.updateCanvas = function() {
		/** TODO: Finish this */
	};
	
	/**
	 * A function that should be called in conjunction with requestionAnimationFrame
	 * to refresh this canvas.  Continually generates callbacks as long as the
	 * audio is playing, but stops generating callbacks when it is paused to save
	 * computation
	 */
	this.repaint = function() {
		this.updateCanvas();
		if (!this.audio_obj.audio_widget.paused) {
			requestAnimationFrame(this.repaint.bind(this));
		}
	};

}