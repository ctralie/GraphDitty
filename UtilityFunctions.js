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