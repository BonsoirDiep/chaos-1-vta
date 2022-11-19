const { parse } = require("csv-parse");

function timex(text, extent){
  var str= '';
  if(extent){
    text= text.replace(/<date/g, '<date val=');
    var $= require('cheerio').load('<body>'+ text+ '</body>');
    if(!$('date').length) str+= '<TIMEX>UNKorNEG</TIMEX>'+ '\n';
    else $('date').each(function(){
      var s2= $(this).text()||'UNKorNEG';
      if($(this).attr('val')== '{}') s2= 'UNKorNEG';
      str+=  '<TIMEX>'+ s2.trim()+ '</TIMEX>'+ '\n';
    });
  } else {
    if(!/<date{/.exec(text)) str+= '<TIMEX>UNKorNEG</TIMEX>'+ '\n';
    else text.replace(/<date{([^\<]*)}>/g, function(s1,s2){
      if(!s2 || s2== '') s2= 'UNKorNEG';
      str+= '<TIMEX>'+ s2.trim()+ '</TIMEX>'+ '\n';
    });
  }

  str+= 'TEXT:\n'+ text;
  return str;
}

function timeArr(arrText, text){
  var str= '';
  var arr= JSON.parse(arrText.replace(/\'/g, '\"'));
  if(!arr.length){
    str+= '<TIMEX>UNKorNEG</TIMEX>'+ '\n';
  }
  else arr.forEach(function(el){
    el= el.trim();
    if(!el) el= 'UNKorNEG';
    str+= '<TIMEX>'+ el+ '</TIMEX>'+ '\n';
  });
  str+= 'TEXT:\n'+ text;
  return str;
}

const { readdirSync, rmSync } = require('fs');
function unsafeRun(directory){
	try{
		readdirSync(directory).forEach(f => rmSync(`${directory}/${f}`));
		require('fs').mkdirSync(directory);
	} catch(err) {
		console.error(err);
	}
}

unsafeRun( __dirname+ '/ann_extent');
unsafeRun( __dirname+ '/sub_extent');

unsafeRun( __dirname+ '/ann_attribute');
unsafeRun( __dirname+ '/sub_attribute');


require('fs').createReadStream(__dirname+ '/result_gg.csv')
.pipe(parse({ delimiter: ",", from_line: 2 }))
.on("data", function (row) {
  // console.log(row);
  var fileN= ''+ (parseInt(row[0])+ 1);

  require('fs').writeFileSync(__dirname+ '/ann_extent/'+ fileN+ '_ann.txt', timex(row[1], 1));
  
  require('fs').writeFileSync(__dirname+ '/sub_extent/'+ fileN+ '_sub.txt', timex(row[2], 1));
  // require('fs').writeFileSync(__dirname+ '/sub_extent/'+ fileN+ '_sub.txt', timeArr(row[3], row[2]));

  require('fs').writeFileSync(__dirname+ '/ann_attribute/'+ fileN+ '_ann.txt', timex(row[1], 0));
  require('fs').writeFileSync(__dirname+ '/sub_attribute/'+ fileN+ '_sub.txt', timex(row[2], 0));
})
.on("end", function () {
  console.log("finished");
})
.on("error", function (error) {
  console.log(error.message);
});
