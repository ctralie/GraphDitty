var gl;
var glcanvas;

function initGL(canvas) {
    try {
        gl = canvas.getContext("experimental-webgl");
        gl.viewportWidth = canvas.width;
        gl.viewportHeight = canvas.height;
    } catch (e) {
    }
    if (!gl) {
        alert("Could not initialise WebGL, sorry :-(.  Try a new version of chrome or firefox and make sure your newest graphics drivers are installed");
    }
}


//Type 0: Fragment shader, Type 1: Vertex Shader
function getShader(gl, str, type) {
    var xmlhhtp;
    var shader;
    if (type == 0) {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    } else if (type == 1) {
        shader = gl.createShader(gl.VERTEX_SHADER);
    } else {
        return null;
    }

    gl.shaderSource(shader, str);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
        return null;
    }

    return shader;
}


var shaderProgram;
var shaderProgram2;

function initShaders() {
    var str = "precision mediump float;\n";
    str = str + "varying vec4 fColor;\n";
    str = str + "void main(void) {\n";
    str = str + "gl_FragColor = fColor;\n";
    str = str + "}\n\n";
    var fragmentShader = getShader(gl, str, 0);
    var fragmentShader2 = getShader(gl, str, 0);

    var strFirst = "attribute vec3 vPos;\n";
    strFirst = strFirst + "attribute vec4 vColor;\n";
    strFirst = strFirst + "uniform mat4 uMVMatrix;\n";
    strFirst = strFirst + "uniform mat4 uPMatrix;\n";
    strFirst = strFirst + "varying vec4 fColor;\n";
    strFirst = strFirst + "void main(void) {\n";

    var strSecond = "gl_Position = uPMatrix * uMVMatrix * vec4(vPos, 1.0);\n";
    strSecond = strSecond+ "fColor = vColor;\n";
    strSecond = strSecond + "}";
    var vertexShader = getShader(gl, strFirst + "gl_PointSize = 3.0;\n" + strSecond, 1);
    var vertexShader2 = getShader(gl, strFirst + "gl_PointSize = 15.0;\n" + strSecond, 1);

    shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }
    gl.useProgram(shaderProgram);
    shaderProgram.vPosAttrib = gl.getAttribLocation(shaderProgram, "vPos");
    gl.enableVertexAttribArray(shaderProgram.vPosAttrib);
    shaderProgram.vColorAttrib = gl.getAttribLocation(shaderProgram, "vColor");
    gl.enableVertexAttribArray(shaderProgram.vColorAttrib);
    shaderProgram.pMatrixUniform = gl.getUniformLocation(shaderProgram, "uPMatrix");
    shaderProgram.mvMatrixUniform = gl.getUniformLocation(shaderProgram, "uMVMatrix");

    shaderProgram2 = gl.createProgram();
    gl.attachShader(shaderProgram2, vertexShader2);
    gl.attachShader(shaderProgram2, fragmentShader2);
    gl.linkProgram(shaderProgram2);
    if (!gl.getProgramParameter(shaderProgram2, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }
    gl.useProgram(shaderProgram2);
    shaderProgram2.vPosAttrib = gl.getAttribLocation(shaderProgram2, "vPos");
    gl.enableVertexAttribArray(shaderProgram2.vPosAttrib);
    shaderProgram2.vColorAttrib = gl.getAttribLocation(shaderProgram2, "vColor");
    gl.enableVertexAttribArray(shaderProgram2.vColorAttrib);
    shaderProgram2.pMatrixUniform = gl.getUniformLocation(shaderProgram2, "uPMatrix");
    shaderProgram2.mvMatrixUniform = gl.getUniformLocation(shaderProgram2, "uMVMatrix");
}


var mvMatrix = mat4.create();
var mvMatrixStack = [];
var pMatrix = mat4.create();
var offset = vec3.create();

function mvPopMatrix() {
    if (mvMatrixStack.length == 0) {
        throw "Invalid popMatrix!";
    }
    mvMatrix = mvMatrixStack.pop();
}


function setUniforms(sP) {
    gl.uniformMatrix4fv(sP.pMatrixUniform, false, pMatrix);
    gl.uniformMatrix4fv(sP.mvMatrixUniform, false, mvMatrix);
    //gl.uniform1f(shaderProgram.pointSizeUniform, false, pointSize);
}


function degToRad(degrees) {
    return degrees * Math.PI / 180;
}

//Mouse variables
var usingMouse = false;
var lastX = 0;
var lastY = 0;
var dragging = false;
var MOUSERATE = 0.005;
var clickType = "LEFT";
var justClicked = false;
var translateRotate = true;

getMousePos = function(evt) {
    return {
        X: evt.clientX,
        Y: evt.clientY
    };
}

releaseClick = function(evt) {
    usingMouse = true;
	evt.preventDefault();
	dragging = false;
	requestAnimFrame(repaint);
	return false;
}

mouseOut = function(evt) {
    usingMouse = true;
	dragging = false;
	requestAnimFrame(repaint);
	return false;
}

makeClick = function(e) {
    usingMouse = true;
    var evt = (e == null ? event:e);
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
	dragging = true;
	justClicked = true;
	var mousePos = getMousePos(evt);
	lastX = mousePos.X;
	lastY = mousePos.Y;
	requestAnimFrame(repaint);
	return false;
}

//http://www.w3schools.com/jsref/dom_obj_event.asp
clickerDragged = function(evt) {
    usingMouse = true;
	evt.preventDefault();
	var mousePos = getMousePos(evt);
	var dX = mousePos.X - lastX;
	var dY = mousePos.Y - lastY;
	lastX = mousePos.X;
	lastY = mousePos.Y;
	if (dragging) {
	    if (clickType == "MIDDLE") {
	        camera.translate(dX, -dY);
	    }
		else if (clickType == "RIGHT") { //Right click
			camera.zoom(dY); //Want to zoom in as the mouse goes up
		}
		else if (clickType == "LEFT") {
		    if (evt.ctrlKey) {
		        camera.translate(dX, -dY);
		    }
		    else if (evt.shiftKey) {
		        camera.zoom(dY);
		    }
		    else{
			    camera.orbitLeftRight(dX);
			    camera.orbitUpDown(-dY);
			}
		}
	    requestAnimFrame(repaint);
	}
	return false;
}

touchDragStart = function(evt) {
    if (usingMouse) {
        return;
    }
    lastX = evt.pageX;
    lastY = evt.pageY;
}

touchDragged = function(evt) {
    if (usingMouse) {
        return;
    }
    var dX = evt.pageX - lastX;
    var dY = evt.pageY - lastY;
    lastX = evt.pageX;
    lastY = evt.pageY;
    if (translateRotate) {
        camera.translate(dX, -dY);
    }
    else {
        camera.orbitLeftRight(dX);
        camera.orbitUpDown(-dY);
    }
    requestAnimFrame(repaint);
}

touchGesture = function(evt) {
    //waitingDisp.innerHTML = "<font color = \"red\">" + evt.toString() + "</font>";
    lastX = evt.pageX;
    lastY = evt.pageY;
    //camera.orbitLeftRight(2*evt.da);
    camera.zoom(-evt.ds*200);
    requestAnimFrame(repaint);
}

touchToggleTap = function(evt) {
    if (usingMouse) {
        return;
    }
    translateRotate = !translateRotate;
}

//Playing information
var playIdx = 0;
var playTime = 0;
var startTime = 0;
var offsetTime = 0;
var playing = false;

//Delay Series Information
var DelaySeries = [ [] ];

//Centers Information
var centers = [];
var dims = [];
var edges = [];

//Vertex/color buffers for the entire point cloud
var allVertexVBO = -1;
var allColorVBO = -1;
var edgesVertexVBO = -1;
var edgesColorVBO = -1;
var times = [];

//Camera stuff
var camera = new MousePolarCamera(800, 600, 0.75);
var farR = 1.0;
var bbox = [0, 1, 0, 1, 0, 1];

//Animation stuff
var capturer = null;
var animFrameNum = 0;
var NAnimFrames = 30;
var origCamera = {};

function centerOnBBox() {
    var dX = bbox[1] - bbox[0];
    var dY = bbox[3] - bbox[2];
    var dZ = bbox[5] - bbox[4];
    farR = Math.sqrt(dX*dX + dY*dY + dZ*dZ);
    camera.R = farR;
    camera.center = vec3.fromValues(bbox[0] + 0.5*dX, bbox[2] + 0.5*dY, bbox[4] + 0.5*dZ);
    camera.phi = Math.PI/2;
    camera.theta = -Math.PI/2;
    camera.updateVecsFromPolar();
}

function initGLBuffers(X) {
    var N = X.length;
    if (N <= 0) {
        return;
    }
    DelaySeries = X;
    playIdx = N-1;
    playTime = X[X.length-1][3];
    var i = 0;
    var k = 0;

    var vertices = [];
    var colors = [];
    times = [];
    var label;
    var dim;

    for (i = 0; i < N; i++) {
    	for (k = 0; k < 3; k++) {
    		vertices.push(X[i][k]);
    	}
    	times.push(X[i][3]);
    	ci = 63.0*(0.1+X[i][3])/(0.1+X[N-1][3]);
    	li = numeric.floor([ci])[0];
    	ri = numeric.ceil([ci])[0];
    	ri = numeric.min([ri], [63])[0];
    	//Linear interpolation for colormap
    	colors.push(COLORMAP_JET[li*3]*(ri-ci) + COLORMAP_JET[ri*3]*(ci-li));//Red
    	colors.push(COLORMAP_JET[li*3+1]*(ri-ci) + COLORMAP_JET[ri*3+1]*(ci-li));//Green
    	colors.push(COLORMAP_JET[li*3+2]*(ri-ci) + COLORMAP_JET[ri*3+2]*(ci-li));//Blue
    	colors.push(1);//Alpha
    }

    //Initialize vertex buffers
    centerCounts = [];
    if (allVertexVBO == -1) {
        allVertexVBO = gl.createBuffer();
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, allVertexVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    allVertexVBO.itemSize = 3;
    allVertexVBO.numItems = N;


    //Initialize color buffers
    if (allColorVBO == -1) {
        allColorVBO = gl.createBuffer();
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, allColorVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    allColorVBO.itemSize = 4;
    allColorVBO.numItems = N;

    //Initialize edge buffers
    if (edgesVertexVBO == -1) {
        edgesVertexVBO = gl.createBuffer();
    }
    if (edgesColorVBO == -1) {
        edgesColorVBO = gl.createBuffer();
    }
    var edgesV = [];
    var edgesC = [];
    var eNum;
    for (i = 0; i < edges.length; i++) {
        for (eNum = 0; eNum < 2; eNum++) {
            for (k = 0; k < 3; k++) {
                edgesV.push(DelaySeries[centers[edges[i][eNum]]][k]);
                edgesC.push(0.8);
            }
            edgesC.push(1);
        }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, edgesVertexVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(edgesV), gl.STATIC_DRAW);
    edgesVertexVBO.itemSize = 3;
    edgesVertexVBO.numItems = edges.length*2;
    gl.bindBuffer(gl.ARRAY_BUFFER, edgesColorVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(edgesC), gl.STATIC_DRAW);
    edgesColorVBO.itemSize = 4;
    edgesColorVBO.numItems = edges.length*2;

    //Now determine the bounding box of the curve and use
    //that to update the camera info
    bbox = [vertices[0], vertices[0], vertices[1], vertices[1], vertices[2], vertices[2]];
    for (i = 0; i < N; i++) {
        if (vertices[i*3] < bbox[0]) {
            bbox[0] = vertices[i*3];
        }
        if (vertices[i*3] > bbox[1]) {
            bbox[1] = vertices[i*3];
        }
        if (vertices[i*3+1] < bbox[2]) {
            bbox[2] = vertices[i*3+1];
        }
        if (vertices[i*3+1] > bbox[3]) {
            bbox[3] = vertices[i*3+1];
        }
        if (vertices[i*3+2] < bbox[4]) {
            bbox[4] = vertices[i*3+2];
        }
        if (vertices[i*3+2] > bbox[5]) {
            bbox[5] = vertices[i*3+2];
        }
    }
    centerOnBBox();
    requestAnimFrame(repaint);
}


function drawScene() {
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    mat4.perspective(pMatrix, 45, gl.viewportWidth / gl.viewportHeight, camera.R/100.0, Math.max(farR*2, camera.R*2));
    mvMatrix = camera.getMVMatrix();

    if (allVertexVBO != -1 && allColorVBO != -1) {
        while (DelaySeries[playIdx][3] < playTime && playIdx < DelaySeries.length - 1) {
            playIdx++;
        }
        gl.useProgram(shaderProgram);
        setUniforms(shaderProgram);
        //Step 1: Draw all points unsaturated
        gl.bindBuffer(gl.ARRAY_BUFFER, allVertexVBO);
        gl.vertexAttribPointer(shaderProgram.vPosAttrib, allVertexVBO.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, allColorVBO);
        gl.vertexAttribPointer(shaderProgram.vColorAttrib, allColorVBO.itemSize, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, playIdx);
        //Draw Lines between points if the user so chooses
        if (MusicParams.displayTimeEdges) {
            gl.drawArrays(gl.LINES, 0, playIdx+1);
            gl.drawArrays(gl.LINES, 1, playIdx);
        }


        //Step 2: Draw the current point as a larger point
        gl.useProgram(shaderProgram2);
        setUniforms(shaderProgram2);
        gl.bindBuffer(gl.ARRAY_BUFFER, allVertexVBO);
        gl.vertexAttribPointer(shaderProgram2.vPosAttrib, allVertexVBO.itemSize, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, allColorVBO);
        gl.vertexAttribPointer(shaderProgram2.vColorAttrib, allColorVBO.itemSize, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, playIdx, 1);
    }
}

function drawSceneAnim() {
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    mat4.perspective(pMatrix, 45, gl.viewportWidth / gl.viewportHeight, camera.R/100.0, Math.max(farR*2, camera.R*2));
    mvMatrix = camera.getMVMatrix();

    if (allVertexVBO != -1 && allColorVBO != -1) {
        gl.useProgram(shaderProgram);
        setUniforms(shaderProgram);
        //Step 1: Draw all points unsaturated
        gl.bindBuffer(gl.ARRAY_BUFFER, allVertexVBO);
        gl.vertexAttribPointer(shaderProgram.vPosAttrib, allVertexVBO.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, allColorVBO);
        gl.vertexAttribPointer(shaderProgram.vColorAttrib, allColorVBO.itemSize, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, DelaySeries.length);
        //Draw Lines between points if the user so chooses
        if (MusicParams.displayTimeEdges) {
            gl.drawArrays(gl.LINES, 0, DelaySeries.length-2);
            gl.drawArrays(gl.LINES, 1, DelaySeries.length-1);
        }
    }
}


function repaint() {
    drawScene();
}

function repaintWithContext(context) {
    if (playing) {
        playTime = context.currentTime - startTime + offsetTime;
        var timeSlider = document.getElementById('timeSlider');
        timeSlider.value = "" + parseInt(""+Math.round(playTime*1000.0/buffer.duration));
        drawScene();
        requestAnimFrame(function(){repaintWithContext(context)});
    }
    else {
        //If paused allow scrolling around
        playTime = offsetTime;
        drawScene();
    }
}

function repaintAnimation() {
    if (animFrameNum < NAnimFrames) {
        camera.theta = 2*Math.PI*animFrameNum/NAnimFrames;
        camera.updateVecsFromPolar();
        drawSceneAnim();
        capturer.addFrame(glcanvas, {copy:true, delay:100});
        animFrameNum++;
        requestAnimFrame(repaintAnimation);
    }
    else {
        loadString = "Making GIF";
        capturer.render();

        //Restore original camera parameters and repaint
        camera.theta = origCamera.theta;
        camera.phi = origCamera.phi;
        camera.R = origCamera.R;
        camera.center = origCamera.center;
        camera.updateVecsFromPolar();
        requestAnimFrame(drawScene);
    }
}

function startAnimation() {
    loading = true;
    loadString = "Rendering frames";
    loadColor = "yellow";
    changeLoad();
    capturer = new GIF({workers:2, quality:10, workerScript:"libs/gif.worker.js"});
    capturer.on('finished', function(blob) {
      window.open(URL.createObjectURL(blob));
      loading = false;
      waitingDisp.innerHTML = "<h3><font color = \"#00FF00\">Finished</font></h3>";
    });
    animFrameNum = 0;
    var C = vec3.create();
    vec3.copy(C, camera.center);
    origCamera = {theta:camera.theta, phi:camera.phi, R:camera.R, center:C};
    centerOnBBox();
    requestAnimFrame(repaintAnimation);
}


function webGLStart() {
    glcanvas = document.getElementById("LoopDittyGLCanvas");
    glcanvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking

    glcanvas.addEventListener('mousedown', makeClick);
    glcanvas.addEventListener('mouseup', releaseClick);
    glcanvas.addEventListener('mousemove', clickerDragged);
    glcanvas.addEventListener('mouseout', mouseOut);

    glcanvas.addEventListener('pointerdown', makeClick);
    glcanvas.addEventListener('pointerup', releaseClick);
    glcanvas.addEventListener('pointermove', clickerDragged);
    glcanvas.addEventListener('pointerout', mouseOut);

    /*interact(glcanvas)
        .draggable({onstart:touchDragStart, onmove:touchDragged})
        .gesturable({onmove:touchGesture})
        .on('doubletap', touchToggleTap)
        .preventDefault('always');*/

    initGL(glcanvas);
    initShaders();
    centers = [0];
    dims = [1];
    initGLBuffers(WELCOME_CURVE);

    gl.clearColor(0, 0, 0, 1.0);
    gl.enable(gl.DEPTH_TEST);

    requestAnimFrame(repaint);
}
