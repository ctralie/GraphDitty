/**
 * A class to show loading progress
 */

function ProgressBar() {
    //A function to show a progress bar
    this.loading = false;
    this.loadString = "Loading";
    this.loadColor = "yellow";
    this.ndots = 0;
    this.waitingDisp = document.getElementById("pagestatus");
    this.changeLoad = function() {
        if (!this.loading) {
            return;
        }
        var s = "<h3><font color = \"" + this.loadColor + "\">" + this.loadString;
        for (var i = 0; i < this.ndots; i++) {
            s += ".";
        }
        s += "</font></h3>";
        this.waitingDisp.innerHTML = s;
        if (this.loading) {
            this.ndots = (this.ndots + 1)%12;
            setTimeout(this.changeLoad.bind(this), 200);
        }
    };
    this.changeToReady = function() {
        this.loading = false;
        this.waitingDisp.innerHTML = "<h3><font color = \"#00FF00\">Ready</font></h3>";
    };
    this.setLoadingFailed = function() {
        this.loading = false;
        this.waitingDisp.innerHTML = "<h3><font color = \"red\">Loading Failed :(</font></h3>";
    };
}