function ClickConnect(){
  document.querySelector("colab-connect-button").click(); 
  console.log("click the button!")
}
setInterval(ClickConnect,1000*60); // "1分ごとにクリック"