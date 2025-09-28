$(function() {
  var oldWidth, oldMargin ,isFullscreenMode=false;
  $("#fullscreen-btn").on("click", function(){

    if(!isFullscreenMode){
      oldWidth = $("#result .slide").css("width");
      oldMargin = $("#result .slide").css("margin");
      $("#result .slide").css({
        "width": "99%",
        "margin": "0 auto"
      })
      $("#result").toggleFullScreen();
      isFullscreenMode = true;
    }else{
      $("#result .slide").css({
        "width": oldWidth,
        "margin": oldMargin
      })
      $("#result").toggleFullScreen();
      isFullscreenMode = false;
    }
  });
  $(document).bind("fullscreenchange", function() {
    if(!$(document).fullScreen()){
      $("#result .slide").css({
        "width": oldWidth,
        "margin": oldMargin
      })
    }
  });
});