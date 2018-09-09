//Programmer: Chris Tralie
//Purpose: To provide a canvas for drawing a self-similarity matrix synchronized
//with the music

function StructureCanvas(audio_obj) {
	this.ssmcanvas = document.getElementById('SimilarityCanvas');
    this.eigcanvas = document.getElementById('EigCanvas');
	this.ssmctx = this.ssmcanvas.getContext('2d');
    this.eigctx = this.eigcanvas.getContext('2d');
	this.CSImage = new Image;
	this.EigImage = new Image;
	this.audio_obj = audio_obj;

	this.updateParams = function(params) {
		this.CSImage.src = params.W;
		this.audio_obj.audio_widget.style.width = this.CSImage.width;
		if ('v' in params) {
			this.EigImage.src = params.v;
		}
		this.offsetidx = 0;
		this.ssmcanvas.width = this.CSImage.width;
		this.ssmcanvas.height = this.CSImage.height;
		this.eigcanvas.width = this.EigImage.width;
		this.eigcanvas.eight = this.EigImage.height;

		requestAnimationFrame(this.repaint.bind(this));
	}

	this.releaseClickSSM = function(evt) {
		evt.preventDefault();
		var offset1idx = evt.offsetY;
		var offset2idx = evt.offsetX;
	
		clickType = "LEFT";
		evt.preventDefault();
		if (evt.which) {
			if (evt.which == 3) clickType = "RIGHT";
			if (evt.which == 2) clickType = "MIDDLE";
		}
		else if (evt.button) {
			if (evt.button == 2) clickType = "RIGHT";
			if (evt.button == 4) clickType = "MIDDLE";
		}
	
		if (clickType == "LEFT") {
			this.audio_obj.offsetidx = offset1idx;
		}
		else {
			this.audio_obj.offsetidx = offset2idx;
		}
		this.audio_obj.audio_widget.currentTime = this.audio_obj.times[this.audio_obj.offsetidx];
		this.repaint();
		return false;
	}
	
	this.releaseClickEig = function(evt) {
		evt.preventDefault();
		this.audio_obj.offsetidx = evt.offsetX;
		this.audio_obj.audio_widget.currentTime = this.audio_obj.times[this.audio_obj.offsetidx];
		this.repaint();
		return false;
	}
	
	this.makeClick = function(evt) {
		evt.preventDefault();
		return false;
	}
	
	this.clickerDragged = function(evt) {
		evt.preventDefault();
		return false;
	}
	
	this.drawCanvas = function() {
		if (!this.CSImage.complete || !this.EigImage.complete) {
			//Keep requesting redraws until the image has actually loaded
			requestAnimationFrame(this.repaint.bind(this));
		}
		else {
			this.ssmctx.drawImage(this.CSImage, 0, 0);
			this.eigctx.drawImage(this.EigImage, 0, 0);
			
			this.ssmctx.beginPath();
			this.ssmctx.moveTo(0, this.audio_obj.offsetidx);
			this.ssmctx.lineTo(this.CSImage.width, this.audio_obj.offsetidx);
			this.ssmctx.moveTo(this.audio_obj.offsetidx, 0);
			this.ssmctx.lineTo(this.audio_obj.offsetidx, this.CSImage.height);
			this.ssmctx.strokeStyle = '#00ffff';
			this.ssmctx.stroke();
			
			this.eigctx.beginPath();
			this.eigctx.moveTo(this.audio_obj.offsetidx, 0);
			this.eigctx.lineTo(this.audio_obj.offsetidx, this.EigImage.height);
			this.eigctx.strokeStyle = '#00ffff';
			this.eigctx.stroke();
		}
	}
	
	this.repaint = function() {
		var t = this.audio_obj.audio_widget.currentTime;
		while (this.audio_obj.times[this.audio_obj.offsetidx] < t && 
				this.audio_obj.offsetidx < this.audio_obj.times.length - 1) {
			this.audio_obj.offsetidx++;
		}
		this.drawCanvas();
		if (!this.audio_obj.audio_widget.paused) {
			requestAnimationFrame(this.repaint.bind(this));
		}
	}

	this.initDummyListeners = function(canvas) {
		canvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking
		canvas.addEventListener('mousedown', this.makeClick.bind(this));
		canvas.addEventListener('mousemove', this.clickerDragged.bind(this));
		canvas.addEventListener('touchstart', this.makeClick.bind(this));
		canvas.addEventListener('touchmove', this.clickerDragged.bind(this));
		canvas.addEventListener('contextmenu', function dummy(e) { return false });
	}
	
	this.initCanvasHandlers = function() {
		this.initDummyListeners(this.ssmcanvas);
		this.ssmcanvas.addEventListener('mouseup', this.releaseClickSSM.bind(this));
		this.ssmcanvas.addEventListener('touchend', this.releaseClickSSM.bind(this));
		this.initDummyListeners(this.eigcanvas);
		this.eigcanvas.addEventListener('mouseup', this.releaseClickEig.bind(this));
		this.eigcanvas.addEventListener('touchend', this.releaseClickEig.bind(this));
	}

	this.initCanvasHandlers();
}