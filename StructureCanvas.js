//Programmer: Chris Tralie
//Purpose: To provide a canvas for drawing a self-similarity matrix synchronized
//with the music
var CSImage = new Image;
var EigImage = new Image;
var ssmctx;
var offset = 0;
var offsetidx = 0;
var startTime = 0;
var offsetTime = 0;
var playIdxSSM = 0;

//Functions to handle mouse motion

function releaseClickSSM(evt) {
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
	    offsetidx = offset1idx;
	}
	else {
	    offsetidx = offset2idx;
	}
    if (playing) {
        source.stop();
        playAudio();
    }
    else {
    	redrawSSMCanvas();
    }
	return false;
}

function releaseClickEig(evt) {
	evt.preventDefault();
	offsetidx = evt.offsetX;
    if (playing) {
        source.stop();
        playAudio();
    }
    else {
    	redrawSSMCanvas();
    }
	return false;
}

function makeClick(evt) {
	evt.preventDefault();
	return false;
}

function clickerDragged(evt) {
	evt.preventDefault();
	return false;
}

function initDummyListeners(canvas) {
    canvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking
    canvas.addEventListener('mousedown', makeClick);
    canvas.addEventListener('mousemove', clickerDragged);
    canvas.addEventListener('touchstart', makeClick);
    canvas.addEventListener('touchmove', clickerDragged);
    canvas.addEventListener('contextmenu', function dummy(e) { return false });
}

function initCanvasHandlers() {
    var canvas = document.getElementById('SimilarityCanvas');
    initDummyListeners(canvas);
    canvas.addEventListener('mouseup', releaseClickSSM);
    canvas.addEventListener('touchend', releaseClickSSM);
    
    canvas = document.getElementById('EigCanvas');
    initDummyListeners(canvas);
    canvas.addEventListener('mouseup', releaseClickEig);
    canvas.addEventListener('touchend', releaseClickEig);
}

function redrawSSMCanvas() {
	if (!CSImage.complete || !EigImage.complete) {
	    //Keep requesting redraws until the image has actually loaded
	    requestAnimationFrame(redrawSSMCanvas);
	}
	else {
		if (!dimsUpdated) {
			dimsUpdated = true;
			ssmcanvas.width = CSImage.width;
            ssmcanvas.height = CSImage.height;
            eigcanvas.width = EigImage.width;
            eigcanvas.eight = EigImage.height;
		}
	    ssmctx.drawImage(CSImage, 0, 0);
	    eigctx.drawImage(EigImage, 0, 0);
	    
	    ssmctx.beginPath();
        ssmctx.moveTo(0, offsetidx);
        ssmctx.lineTo(CSImage.width, offsetidx);
        ssmctx.moveTo(offsetidx, 0);
        ssmctx.lineTo(offsetidx, CSImage.height);
	    ssmctx.strokeStyle = '#00ffff';
	    ssmctx.stroke();
	    
	    eigctx.beginPath();
	    eigctx.moveTo(offsetidx, 0);
	    eigctx.lineTo(offsetidx, EigImage.height);
	    eigctx.strokeStyle = '#00ffff';
	    eigctx.stroke();
    }
}

function updateSSMCanvas() {
	var t = context.currentTime - startTime + offsetTime;
	timeTxt.innerHTML = Number.parseFloat(t).toFixed(2) + " sec";

	while (times[playIdxSSM] < t && playIdxSSM < times.length - 1) {
		playIdxSSM++;
	}
	offsetidx = playIdxSSM;
	redrawSSMCanvas();
	if (playing) {
		requestAnimationFrame(updateSSMCanvas);
	}
}
