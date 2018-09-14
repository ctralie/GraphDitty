function degToRad(degrees) {
    return degrees * Math.PI / 180;
}


function getMousePos(evt) {
    return {
        X: evt.clientX,
        Y: evt.clientY
    };
}

function DiffusionGLCanvas(audio_obj) {
    this.audio_obj = audio_obj;
    this.glcanvas = document.getElementById("DiffusionGLCanvas");
    this.gl = 0;
    
    // Transformation matrices and camera stuff
    this.mvMatrix = mat4.create();
    this.mvMatrixStack = [];
    this.pMatrix = mat4.create();
    this.offset = vec3.create();
    this.camera = new MousePolarCamera(800, 600, 0.75);
    this.farR = 1.0;
    this.bbox = [0, 1, 0, 1, 0, 1];

    //Mouse variables
    this.usingMouse = false;
    this.lastX = 0;
    this.lastY = 0;
    this.dragging = false;
    this.MOUSERATE = 0.005;
    this.clickType = "LEFT";
    this.justClicked = false;
    this.translateRotate = true;

    //Vertex/color buffers for the entire point cloud
    this.N = 0; //Number of points
    this.allVertexVBO = -1;
    this.allColorVBO = -1;
    this.times = [];

    //Animation stuff
    this.capturer = null;
    this.animFrameNum = 0;
    this.NAnimFrames = 30;
    this.origCamera = {};



    /*******************************************************/
    /*                 CAMERA FUNCTIONS                    */
    /*******************************************************/
    this.mvPopMatrix = function() {
        if (this.mvMatrixStack.length == 0) {
            throw "Invalid popMatrix!";
        }
        this.mvMatrix = this.mvMatrixStack.pop();
    };
    
    
    this.setUniforms = function(sP) {
        this.gl.uniformMatrix4fv(sP.pMatrixUniform, false, this.pMatrix);
        this.gl.uniformMatrix4fv(sP.mvMatrixUniform, false, this.mvMatrix);
        //this.gl.uniform1f(this.shaderProgram.pointSizeUniform, false, pointSize);
    };


    this.centerOnBBox = function() {
        var dX = this.bbox[1] - this.bbox[0];
        var dY = this.bbox[3] - this.bbox[2];
        var dZ = this.bbox[5] - this.bbox[4];
        this.farR = Math.sqrt(dX*dX + dY*dY + dZ*dZ);
        this.camera.R = this.farR;
        this.camera.center = vec3.fromValues(this.bbox[0] + 0.5*dX, this.bbox[2] + 0.5*dY, this.bbox[4] + 0.5*dZ);
        this.camera.phi = Math.PI/2;
        this.camera.theta = -Math.PI/2;
        this.camera.updateVecsFromPolar();
    };
    






    /*******************************************************/
    /*                  MOUSE FUNCTIONS                    */
    /*******************************************************/

    this.releaseClick = function(evt) {
        this.usingMouse = true;
        evt.preventDefault();
        this.dragging = false;
        requestAnimationFrame(this.repaint.bind(this));
        return false;
    };
    
    this.mouseOut = function(evt) {
        this.usingMouse = true;
        this.dragging = false;
        requestAnimationFrame(this.repaint.bind(this));
        return false;
    };
    
    this.makeClick = function(e) {
        this.usingMouse = true;
        var evt = (e == null ? event:e);
        this.clickType = "LEFT";
        evt.preventDefault();
        if (evt.which) {
            if (evt.which == 3) this.clickType = "RIGHT";
            if (evt.which == 2) this.clickType = "MIDDLE";
        }
        else if (evt.button) {
            if (evt.button == 2) this.clickType = "RIGHT";
            if (evt.button == 4) this.clickType = "MIDDLE";
        }
        this.dragging = true;
        this.justClicked = true;
        var mousePos = getMousePos(evt);
        this.lastX = mousePos.X;
        this.lastY = mousePos.Y;
        requestAnimationFrame(this.repaint.bind(this));
        return false;
    };
    
    //http://www.w3schools.com/jsref/dom_obj_event.asp
    this.clickerDragged = function(evt) {
        this.usingMouse = true;
        evt.preventDefault();
        var mousePos = getMousePos(evt);
        var dX = mousePos.X - this.lastX;
        var dY = mousePos.Y - this.lastY;
        this.lastX = mousePos.X;
        this.lastY = mousePos.Y;
        if (this.dragging) {
            if (this.clickType == "MIDDLE") {
                this.camera.translate(dX, -dY);
            }
            else if (this.clickType == "RIGHT") { //Right click
                this.camera.zoom(dY); //Want to zoom in as the mouse goes up
            }
            else if (this.clickType == "LEFT") {
                if (evt.ctrlKey) {
                    this.camera.translate(dX, -dY);
                }
                else if (evt.shiftKey) {
                    this.camera.zoom(dY);
                }
                else{
                    this.camera.orbitLeftRight(dX);
                    this.camera.orbitUpDown(-dY);
                }
            }
            requestAnimationFrame(this.repaint.bind(this));
        }
        return false;
    };
    
    this.touchDragStart = function(evt) {
        if (this.usingMouse) {
            return;
        }
        this.lastX = evt.pageX;
        this.lastY = evt.pageY;
    };
    
    this.touchDragged = function(evt) {
        if (this.usingMouse) {
            return;
        }
        var dX = evt.pageX - lastX;
        var dY = evt.pageY - lastY;
        this.lastX = evt.pageX;
        this.lastY = evt.pageY;
        if (this.translateRotate) {
            this.camera.translate(dX, -dY);
        }
        else {
            this.camera.orbitLeftRight(dX);
            this.camera.orbitUpDown(-dY);
        }
        requestAnimationFrame(this.repaint.bind(this));
    };
    
    this.touchGesture = function(evt) {
        this.lastX = evt.pageX;
        this.lastY = evt.pageY;
        //camera.orbitLeftRight(2*evt.da);
        this.camera.zoom(-evt.ds*200);
        requestAnimationFrame(this.repaint.bind(this));
    };
    
    this.touchToggleTap = function(evt) {
        if (this.usingMouse) {
            return;
        }
        this.translateRotate = !this.translateRotate;
    };

    this.initInteractionHandlers = function() {
        this.glcanvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking
        
        this.glcanvas.addEventListener('mousedown', this.makeClick.bind(this));
        this.glcanvas.addEventListener('mouseup', this.releaseClick.bind(this));
        this.glcanvas.addEventListener('mousemove', this.clickerDragged.bind(this));
        this.glcanvas.addEventListener('mouseout', this.mouseOut.bind(this));

        this.glcanvas.addEventListener('pointerdown', this.makeClick.bind(this));
        this.glcanvas.addEventListener('pointerup', this.releaseClick.bind(this));
        this.glcanvas.addEventListener('pointermove', this.clickerDragged.bind(this));
        this.glcanvas.addEventListener('pointerout', this.mouseOut.bind(this));
    };






    /*******************************************************/
    /*      WEBGL INITIALIZATION / SHADER FUNCTIONS        */
    /*******************************************************/


    this.initGL = function() {
        try {
            this.gl = this.glcanvas.getContext("experimental-webgl");
            this.gl.viewportWidth = this.glcanvas.width;
            this.gl.viewportHeight = this.glcanvas.height;
        } catch (e) {
        }
        if (!this.gl) {
            alert("Could not initialise WebGL, sorry :-(.  Try a new version of chrome or firefox and make sure your newest graphics drivers are installed");
            return;
        }
        this.gl.clearColor(0, 0, 0, 1.0);
        this.gl.enable(this.gl.DEPTH_TEST);
    };

    
    /** Helper function for compiling shaders from strings
     * @param {string} str: The glsl code as a string
     * @param {int} type: 0 if this is a fragment shader, 1 if it's a vertex shader
     */
    this.getShader = function(str, type) {
        var xmlhhtp;
        var shader;
        if (type == 0) {
            shader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        } else if (type == 1) {
            shader = this.gl.createShader(this.gl.VERTEX_SHADER);
        } else {
            return null;
        }

        this.gl.shaderSource(shader, str);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            alert(this.gl.getShaderInfoLog(shader));
            return null;
        }

        return shader;
    }

    
    /**
     * Setup the shaders for drawing the colored curves
     * Create two different shaders whose only difference is the point size
     * (TODO: There's a better way to do this with uniforms)
     */
    this.initShaders = function() {
        var str = "precision mediump float;\n";
        str = str + "varying vec4 fColor;\n";
        str = str + "void main(void) {\n";
        str = str + "gl_FragColor = fColor;\n";
        str = str + "}\n\n";
        var fragmentShader = this.getShader(str, 0);
        var fragmentShader2 = this.getShader(str, 0);
    
        var strFirst = "attribute vec3 vPos;\n";
        strFirst = strFirst + "attribute vec4 vColor;\n";
        strFirst = strFirst + "uniform mat4 uMVMatrix;\n";
        strFirst = strFirst + "uniform mat4 uPMatrix;\n";
        strFirst = strFirst + "varying vec4 fColor;\n";
        strFirst = strFirst + "void main(void) {\n";
    
        var strSecond = "gl_Position = uPMatrix * uMVMatrix * vec4(vPos, 1.0);\n";
        strSecond = strSecond+ "fColor = vColor;\n";
        strSecond = strSecond + "}";
        var vertexShader = this.getShader(strFirst + "gl_PointSize = 3.0;\n" + strSecond, 1);
        var vertexShader2 = this.getShader(strFirst + "gl_PointSize = 15.0;\n" + strSecond, 1);
    
        this.shaderProgram = this.gl.createProgram();
        this.gl.attachShader(this.shaderProgram, vertexShader);
        this.gl.attachShader(this.shaderProgram, fragmentShader);
        this.gl.linkProgram(this.shaderProgram);
        if (!this.gl.getProgramParameter(this.shaderProgram, this.gl.LINK_STATUS)) {
            alert("Could not initialise shaders");
        }
        this.gl.useProgram(this.shaderProgram);
        this.shaderProgram.vPosAttrib = this.gl.getAttribLocation(this.shaderProgram, "vPos");
        this.gl.enableVertexAttribArray(this.shaderProgram.vPosAttrib);
        this.shaderProgram.vColorAttrib = this.gl.getAttribLocation(this.shaderProgram, "vColor");
        this.gl.enableVertexAttribArray(this.shaderProgram.vColorAttrib);
        this.shaderProgram.pMatrixUniform = this.gl.getUniformLocation(this.shaderProgram, "uPMatrix");
        this.shaderProgram.mvMatrixUniform = this.gl.getUniformLocation(this.shaderProgram, "uMVMatrix");
    
        this.shaderProgram2 = this.gl.createProgram();
        this.gl.attachShader(this.shaderProgram2, vertexShader2);
        this.gl.attachShader(this.shaderProgram2, fragmentShader2);
        this.gl.linkProgram(this.shaderProgram2);
        if (!this.gl.getProgramParameter(this.shaderProgram2, this.gl.LINK_STATUS)) {
            alert("Could not initialise shaders");
        }
        this.gl.useProgram(this.shaderProgram2);
        this.shaderProgram2.vPosAttrib = this.gl.getAttribLocation(this.shaderProgram2, "vPos");
        this.gl.enableVertexAttribArray(this.shaderProgram2.vPosAttrib);
        this.shaderProgram2.vColorAttrib = this.gl.getAttribLocation(this.shaderProgram2, "vColor");
        this.gl.enableVertexAttribArray(this.shaderProgram2.vColorAttrib);
        this.shaderProgram2.pMatrixUniform = this.gl.getUniformLocation(this.shaderProgram2, "uPMatrix");
        this.shaderProgram2.mvMatrixUniform = this.gl.getUniformLocation(this.shaderProgram2, "uMVMatrix");
    };
















    /*******************************************************/
    /*           RENDERING/UPDATING FUNCTIONS              */
    /*******************************************************/

    this.updateParams = function(params) {
        if (!this.gl) {
            alert("Error: GL not properly initialized, so cannot display new song");
            return;
        }
        this.N = params.X.length/3;
        if (this.N <= 0) {
            return;
        }
        var i = 0;
        
        //Initialize vertex buffers
        if (this.allVertexVBO == -1) {
            this.allVertexVBO = this.gl.createBuffer();
        }
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.allVertexVBO);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(this.X), this.gl.STATIC_DRAW);
        this.allVertexVBO.itemSize = 3;
        this.allVertexVBO.numItems = this.N;
    
        //Initialize color buffers
        if (this.allColorVBO == -1) {
            this.allColorVBO = this.gl.createBuffer();
        }
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.allColorVBO);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(this.colors), this.gl.STATIC_DRAW);
        this.allColorVBO.itemSize = 4; //Including alpha
        this.allColorVBO.numItems = this.N;
    
        //Now determine the bounding box of the curve and use
        //that to update the camera info
        bbox = [params.X[0], params.X[0], params.X[1], params.X[1], params.X[2], params.X[2]];
        for (i = 0; i < this.N; i++) {
            if (params.X[i*3] < bbox[0]) {
                bbox[0] = params.X[i*3];
            }
            if (params.X[i*3] > bbox[1]) {
                bbox[1] = params.X[i*3];
            }
            if (params.X[i*3+1] < bbox[2]) {
                bbox[2] = params.X[i*3+1];
            }
            if (params.X[i*3+1] > bbox[3]) {
                bbox[3] = params.X[i*3+1];
            }
            if (params.X[i*3+2] < bbox[4]) {
                bbox[4] = params.X[i*3+2];
            }
            if (params.X[i*3+2] > bbox[5]) {
                bbox[5] = params.X[i*3+2];
            }
        }
        this.centerOnBBox();
        requestAnimationFrame(this.repaint.bind(this));
    };
    

	/**
	 * A function which toggles all of the visible elements to show
	 */
	this.show = function() {
		this.glcanvas.style.display = "block";
	};

	/**
	 * A function which toggles all of the visible elements to hide
	 */
	this.hide = function() {
		this.glcanvas.style.display = "none";
	};

    
    this.drawScene = function(making_GIF) {
        this.gl.viewport(0, 0, this.gl.viewportWidth, this.gl.viewportHeight);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
    
        mat4.perspective(this.pMatrix, 45, this.gl.viewportWidth / this.gl.viewportHeight, this.camera.R/100.0, Math.max(this.farR*2, this.camera.R*2));
        this.mvMatrix = this.camera.getMVMatrix();
        
        /** If we're making the GIF, show the whole thing.  If not, we should
         * figure out what index we're at in the song
         */
        var playIdx = this.N;
        if (!making_GIF) {
            var playIdx = this.audio_obj.audio_widget.currentTime / this.audio_obj.time_interval;
            playIdx = Math.round(playIdx);
        }

        if (this.allVertexVBO != -1 && this.allColorVBO != -1) {
            this.gl.useProgram(this.shaderProgram);
            this.setUniforms(this.shaderProgram);
            //Step 1: Draw all points unsaturated
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.allVertexVBO);
            this.gl.vertexAttribPointer(this.shaderProgram.vPosAttrib, this.allVertexVBO.itemSize, this.gl.FLOAT, false, 0, 0);
    
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.allColorVBO);
            this.gl.vertexAttribPointer(this.shaderProgram.vColorAttrib, this.allColorVBO.itemSize, this.gl.FLOAT, false, 0, 0);
            this.gl.drawArrays(this.gl.POINTS, 0, playIdx);
            //Draw "time edge" lines between points
            this.gl.drawArrays(this.gl.LINES, 0, playIdx+1);
            this.gl.drawArrays(this.gl.LINES, 1, playIdx);
    
            if (!making_GIF) {
                //Step 2: Draw the current point as a larger point
                this.gl.useProgram(this.shaderProgram2);
                this.setUniforms(this.shaderProgram2);
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.allVertexVBO);
                this.gl.vertexAttribPointer(this.shaderProgram2.vPosAttrib, this.allVertexVBO.itemSize, this.gl.FLOAT, false, 0, 0);
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.allColorVBO);
                this.gl.vertexAttribPointer(this.shaderProgram2.vColorAttrib, this.allColorVBO.itemSize, this.gl.FLOAT, false, 0, 0);
                this.gl.drawArrays(this.gl.POINTS, playIdx, 1);
            }
        }
    };
    
    
    this.repaint = function() {
        this.drawScene(false);
		if (!this.audio_obj.audio_widget.paused) {
			requestAnimationFrame(this.repaint.bind(this));
		}
    };

    /*
    function repaintAnimation() {
        if (animFrameNum < NAnimFrames) {
            camera.theta = 2*Math.PI*animFrameNum/NAnimFrames;
            camera.updateVecsFromPolar();
            drawSceneAnim();
            capturer.addFrame(glcanvas, {copy:true, delay:100});
            animFrameNum++;
            requestAnimationFrame(repaintAnimation);
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
            requestAnimationFrame(drawScene);
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
        requestAnimationFrame(repaintAnimation);
    }*/
    
    //Now that everying is defined, run the initalization
    this.initInteractionHandlers();
    this.initGL();
    this.initShaders();
    requestAnimationFrame(this.repaint.bind(this));
}