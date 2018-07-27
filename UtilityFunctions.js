//Programmer: Chris Tralie


//A function to show a progress bar
var loading = false;
var loadString = "Loading";
var loadColor = "yellow";
var ndots = 0;
function changeLoad() {
    if (!loading) {
        return;
    }
    var s = "<h3><font color = \"" + loadColor + "\">" + loadString;
    for (var i = 0; i < ndots; i++) {
        s += ".";
    }
    s += "</font></h3>";
    waitingDisp.innerHTML = s;
    if (loading) {
        ndots = (ndots + 1)%12;
        setTimeout(changeLoad, 200);
    }
}
function changeToReady() {
    loading = false;
    waitingDisp.innerHTML = "<h3><font color = \"#00FF00\">Ready</font></h3>";
}
function setLoadingFailed() {
    loading = false;
    waitingDisp.innerHTML = "<h3><font color = \"red\">Loading Failed :(</font></h3>";
}

//Base64 Functions
//http://stackoverflow.com/questions/21797299/convert-base64-string-to-arraybuffer
function base64ToArrayBuffer(base64) {
    var binary =  window.atob(base64);
    var len = binary.length;
    var bytes = new Uint8Array( len );
    for (var i = 0; i < len; i++)        {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
}

function ArrayBufferTobase64(arrayBuff) {
    var binary = '';
    var bytes = new Uint8Array(arrayBuff);
    var N = bytes.byteLength;
    for (var i = 0; i < N; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}
