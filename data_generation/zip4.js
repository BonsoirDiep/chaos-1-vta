var zip3= require('./zip3_cont');
const DEBUG= 0;
function itToTuan(val){
    switch (val) {
        case 'ANNY':
            res= '10X'
            break;
        case '10X':
            res= '100X';
            break;
        case 'HALF':
            res= '4';
            break;
        case '10':
            res= '70';
            break;
        default:
            res= val;
            break;
    }
    return res;
}

// 'ANNY', '10X', 'HALF', '10', '100', '1M'

function itToThapKy(val){
    switch (val) {
        case 'ANNY':
            res= '10X'
            break;
        case '10X':
            res= '100X';
            break;
        case 'HALF':
            res= '5';
            break;
        case '10':
            res= '100';
            break;
        default:
            res= val;
            break;
    }
    return res;
}

function itToTheKy(val){
    switch (val) {
        case 'ANNY':
            res= '100X'
            break;
        case '10X':
            res= '1000X';
            break;
        case 'HALF':
            res= '50';
            break;
        case '10':
            res= '1000';
            break;
        default:
            res= val;
            break;
    }
    return res;
}

function updateDauMinus0(obj){
    obj= obj || {};
    return /(cách đây |cách nay |trước đây |trước )/.exec(obj.prev)? '-': (
        '*'
    );
}
function rdx100(){
    return Math.floor(Math.random()*100);
}

const romans = require('romans');
function romanNumerals(numberX){
    return {tnt: romans.romanize(numberX), num: numberX};
}

function humanNumSpeech(num, haveCommon){
    const xCommon= haveCommon? ',': '';
    if(typeof(num)!='number') return undefined;
    if(!isFinite(num)) return undefined;
    var x= ''+ num, y= '';
    if(x.length> 12) return humanNum(num);
    const so= {
        0: '', 1: 'một', 2: 'hai', 3: 'ba', 4: 'bốn', 5: 'năm', 6: 'sáu', 7: 'bảy', 8: 'tám', 9: 'chín'
    }
    function test(tmp){
        return 'undefined' == typeof tmp ? 0 : ('0'==tmp) ? 0 : !0 ;
    }
    var leOrLink= rdx100()> 25 ? 'lẻ': 'linh';
    for(var i=0; i<x.length; i++){
        switch ((x.length-i)%3) {
            case 0:
                y+= (x[i]==0) ? ( ( !test(x[i+2]) && !test(x[i+1]) ) ? '': 'không trăm ' ) : so[x[i]] + ' trăm ';
                //y+= (x[i]==0) ? '' : so[x[i]] + ' trăm ';
                break;
            case 2:
                y+= (x[i]==1) ? 'mười ' : ( (x[i]==0) ? (x[i+1]==0 ? '': `${leOrLink} `) : so[x[i]] + ' mươi ' );
                break;
            default:
                y+= (x[i]==0) ? '': ( (x[i]==1)? ( (x[i-1]==1 || !x[i-1]) ? 'một ': y.endsWith(`${leOrLink} `)? 'một ': 'mốt ' ): (
                        (x[i]==5)? ( y.endsWith(`${leOrLink} `)||y==''? 'năm ': 'lăm ') : so[x[i]]+ ' '
                    )
                );
                break;
        }
        if((x.length-i)%3===1 && i!==(x.length-1)) {
            if( !test(x[i-2]) && !test(x[i-1]) && !test(x[i]) ) {
                continue;
            }
            switch (x.length-i) {
                case 4:
                    y+= `${rdx100()> 25? 'nghìn': 'ngàn'}${xCommon} `;
                    break;
                case 7:
                    y+= `triệu${xCommon} `;
                    break;
                case 10:
                    y+= `tỷ${xCommon} `;
                    break;
                default:
                    break;
            }
        }
    }

    var tnt= (y[0]+ y.substring(1)+ '').replace(/, $/, '').trim();

    if(rdx100()> 75) tnt= tnt.replace(' mươi ', ' ');

    // if(rdx100()> 75) tnt= tnt.replace(' mươi ', ' ');
    if(num>9 && num<100 && rdx100()<30) tnt= tnt.replace(/ mươi$/, ' chục');

    return {
        tnt,
        num
    };

}

function humanNumText(num){
    if(rdx100()> 30) return {tnt: ''+ num, num};
    var x= ''+ num, y= '';

    function test(tmp){
        return 'undefined' == typeof tmp ? 0 : ('0'==tmp) ? 0 : !0 ;
    }
    var dotOrDox= rdx100()> 25 ? '.': ',';
    for(var i=0; i<x.length; i++){
        y+= x[i];
        if((x.length-i)%3===1 && i!==(x.length-1)) {
            y+= dotOrDox;
        }
    }
    return {
        tnt: y,
        num
    };

}
var a= {
    "{{year}}": function(){
        // "_1991_",
        var t= [];
        var tmp= [];
        for(var i=1200; i<2030;i++){
            tmp= [''+ i, 'NAMX '+ i];
            t.push(tmp);
        }
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    "{{yearxx}}": function(){
        // "_91_ _60_ _ngoái_",
        if(rdx100()>96) {
            return {
                tnt: 'ngoái',
                num: 'NAMX -1'
            };
        }
        var t= [];
        var tmp= [];
        for(var i=10; i<99;i++){
            tmp= [''+ i, 'NAMX 19'+ i];
            t.push(tmp);
        }
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    "{{mon}}": function(){
        // "_01_ _1_ _12_ _mười_ _năm_ _sáu_ _bốn_ _tư_ _giêng_ _chạp_",
        var t= [];
        var tmp= [];
        if(rdx100()> 30) for(var i=1; i<=12;i++){
            tmp= [rdx100()> 65?(''+i).padStart(2,'0'):''+ i, 'THGX '+ i];
            t.push(tmp);
        } else for(var i=1; i<=12;i++){
            tmp= [
                i== 1 && rdx100()> 65 ? 'giêng' : (
                    i== 12 && rdx100()> 65 ? 'chạp' :humanNumSpeech(i).tnt
                ), 'THGX '+ i
            ];
            t.push(tmp);
        }
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    }, 
    "{{day}}": function(){
        // "_01_ _1_ _30_ _31_ _hai mươi lăm_ _mai_ _kia_",
        // return {tnt: 'REF', num: 1}

        if(rdx100()>96) {
            if(rdx100()>50) return {
                tnt: 'mai',
                num: 'NGYX +1'
            }; else return {
                tnt: 'kia',
                num: 'NGYX +2'
            };
            // kia kìa
        }
        var t= [];
        var tmp= [];
        if(rdx100()> 30) for(var i=1; i<32;i++){
            tmp= [rdx100()> 65?(''+i).padStart(2,'0'):''+ i, 'NGYX '+ i];
            t.push(tmp);
        } else for(var i=1; i<32;i++){
            tmp= [humanNumSpeech(i).tnt, 'NGYX '+ i];
            t.push(tmp);
        }
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    "{{centuryxx}}": function(){
        // _10_ _11_ _10 trước công nguyên_ _10 tcn_
        if(rdx100()>96) {
            if(rdx100()>50) return {
                tnt: 'trước',
                num: 'NAMX -100'
            }; else return {
                tnt: 'sau',
                num: 'NAMX +100'
            };
            // kia kìa
        }
        var t= [];
        var tmp= [];
        if(rdx100()> 60) for(var i=1; i<32;i++){
            tmp= [
                ((rdx100()> 85&& i< 11) ? 'thứ ': '') +  (rdx100()> 65? (''+i).padStart(2,'0'):''+ i),
                'NAMX '+ ((i-1)*100+1)
            ];
            t.push(tmp);
        } else for(var i=1; i<32;i++){
            tmp= [rdx100()> 35? romanNumerals(i).tnt: humanNumSpeech(i).tnt, 'NAMX '+ ((i-1)*100+1)];
            t.push(tmp);
        }
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    "{{numberic}}": function(){
        // "_8_ năm sau  _2.000_ năm sau",
        // return {tnt: 'REF', num: 1}
        var t= [];
        var tmp= [];
        var lol= [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000];
        if(rdx100()> 30) for(var i= 1;i<31;i++){
            tmp= [humanNumText(i).tnt, i];
            t.push(tmp);
        }
        else lol.forEach(function(el){
            tmp= [humanNumText(el).tnt, el];
            t.push(tmp);
        })


        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    // "{{decade}}": "thập niên _80_",

    "{{mothai}}": function(){
        // "_một_, _hai_, _năm_, _chục_, _trăm_, _ngàn_, _nghìn_, _triệu_ .. => số lượng thuần túy",
        // return {tnt: 'REF', num: 1}
        var t= [];
        var tmp= [];
        var lol= [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000];
        if(rdx100()> 50) for(var i= 1;i<31;i++){
            tmp= [humanNumSpeech(i).tnt, i];
            t.push(tmp);
        }
        else lol.forEach(function(el){
            tmp= [humanNumSpeech(el).tnt, el];
            t.push(tmp);
        })


        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },

    "{{sangNext}}": function(){
        var t= [
            ['sang', 1]
        ];
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    "{{it}}": function(){
        
        var t= [
            ['ít', 'ANNY'],
            ['mấy mươi', '10X'],
            ['hàng chục', '10X'],
            ['nhiều', 'ANNY'],
            ['mấy', 'ANNY'],
            ['nửa', 'HALF'],
            ['một vài', 'ANNY'],
            ['chục', '10'],
            ['hàng trăm', '100'],
            ['hàng triệu', '1M']
        ];


        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    

    // "{{thutrongtuan}}": function(){
    //     var t= [
    //         ['thứ hai'],
    //         ['thú 2'],
    //         ['thứ ba'],
    //         ['thứ 3'],
    //         ['thứ tư'],
    //         ['thứ 4'],
    //         ['thứ năm'],
    //         ['thứ 5'],
    //         ['thứ sáu'],
    //         ['thứ 6'],
    //         ['thứ bảy'],
    //         ['thứ 7'],
    //         ['chủ nhật'],
    //         ['cn']
    //     ]
    //     return t[Math.floor(Math.random()*t.length)];
    // },
    
    // tùy biến "{{namtgngay}}"
    "{{tuanngay}}": function(prevObj){
        var prevNum= 0;
        // console.log({prevNum});
        var prevDau= '*';
        if(prevObj){
            prevNum= typeof(prevObj.num)=='number'? prevObj.num*7: itToTuan(prevObj.num);
            if(DEBUG) console.log(prevObj, '!!!', typeof(prevObj.num));
            // process.exit();
            // console.log({prevNum});
            prevDau= updateDauMinus0(prevObj);
        }
        var t= [
            ['tuần',    'NGYX '+ prevDau+ prevNum],
            // ['thế kỷ'],
            // ['thập kỷ']
        ];
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    "{{theky}}": function(prevObj){
        var prevNum= 0;
        var prevDau= '*';
        if(prevObj){
            prevNum= typeof(prevObj.num)=='number'? prevObj.num*100: itToTheKy(prevObj.num);
            prevDau= updateDauMinus0(prevObj);
        }
        var t= [
            ['thế kỷ',    'NAMX '+ prevDau+ prevNum]
        ];
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    "{{thapky}}": function(prevObj){
        var prevNum= 0;
        var prevDau= '*';
        if(prevObj){
            prevNum= typeof(prevObj.num)=='number'? prevObj.num*10: itToThapKy(prevObj.num);
            prevDau= updateDauMinus0(prevObj);
        }
        var t= [
            ['thập kỷ',    'NAMX '+ prevDau+ prevNum]
        ];
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    "{{namtgngay}}": function(prevObj){
        var prevNum= 0;
        // console.log({prevNum});
        var prevDau= '*';
        if(prevObj){
            prevNum= prevObj.num;
            if(DEBUG) console.log('{{namtgngay}} -> ', prevObj);
            prevDau= updateDauMinus0(prevObj);
            if(DEBUG)  console.log({prevDau})
        }
        var t= [
            ['năm',     'NAMX '+ prevDau+ prevNum],
            ['tháng',   'THGX '+ prevDau+ prevNum],

            ['ngày',    'NGYX '+ prevDau+ prevNum],
            ['hôm',     'NGYX '+ prevDau+ prevNum]
        ];
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    '{{nayngoaisau}}': function(){// 	=> năm nay|năm ngoái|năm sau'
        var t= [
            ['năm nay', 'NAMX +0'],
            ['năm ngoái', 'NAMX -1'],
            ['năm sau', 'NAMX +1'],
            ['năm trước', 'NAMX -1'],
            ['năm vừa rồi', 'NAMX -1'],
            ['năm vừa qua', 'NAMX -1']
        ]
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    '{{maimot}}': function(){// 	=> tối mai, tối qua
        var t= [
            ['mai', 'NGYX +1'],
            ['qua', 'NGYX -1'],
            ['vừa qua', 'NGYX -1']
        ]
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    },
    '{{truocsau}}': function(){// 	=> tháng trước|tháng sau
        var t= [
            ['trước', 'THGX -1'],
            ['sau', 'THGX +1'],
            ['vừa rồi', 'THGX -1'],
            ['vừa qua', 'THGX -1']
        ]
        var rnd= Math.floor(Math.random()*t.length);
        return {
            tnt: t[rnd][0],
            num: t[rnd][1]
        };
    }
}

var aa= {
    '{{sangNext}}': a['{{sangNext}}'],

    '{{nayngoaisau}}': a['{{nayngoaisau}}'],
    '{{maimot}}': a['{{maimot}}'],
    '{{truocsau}}': a['{{truocsau}}'],

    '{{day}}': a['{{day}}'],
    '{{mon}}': a['{{mon}}'],
    '{{yearxx}}': a['{{yearxx}}'],
    '{{year}}': a['{{year}}'],
    '{{centuryxx}}': a['{{centuryxx}}'],
    
    '{{mothai}}': a['{{mothai}}'],
    '{{numberic}}': a['{{numberic}}'],
    '{{it}}': a['{{it}}'],

    '{{tuanngay}}': a['{{tuanngay}}'],
    '{{thapky}}': a['{{thapky}}'],
    '{{theky}}': a['{{theky}}'],
    '{{namtgngay}}': a['{{namtgngay}}']
}


function crack(text){
    // console.log('>>text:', text);
    var b= {
        template: text,
        reg: []
    };
    var result= {};
    function fixQuaNayTruoc(obj){
        if(DEBUG) console.log('>>> fixQuaNayTruoc', obj);
        if(/(nay|qua|trước)/.exec(obj.next)) {
            if(obj.tnt.includes('tuần')){
                obj.num= obj.num.replace(' *0', ' -7');
            } else if(obj.tnt.includes('thế kỷ')){
                obj.num= obj.num.replace(' *0', ' -100');
            } else if(obj.tnt.includes('thập kỷ')){
                obj.num= obj.num.replace(' *0', ' -10');
            }
            else {
                obj.num= obj.num.replace(' *0', ' -1');
            }
        }
        else if(/(tới|sau|tiếp|nữa)/.exec(obj.next)) {
            if(obj.tnt.includes('tuần')){
                obj.num= obj.num.replace(' *0', ' +7');
            } else if(obj.tnt.includes('thế kỷ')){
                obj.num= obj.num.replace(' *0', ' +100');
            } else if(obj.tnt.includes('thập kỷ')){
                obj.num= obj.num.replace(' *0', ' +10');
            } else {
                obj.num= obj.num.replace(' *0', ' +1');
            }
        }
    }
    Object.keys(aa).forEach(function(key){
        // console.log({key})
        b.template= b.template.replace(key, function(s1){
            var prevObj= b.reg[b.reg.length-1];
            var x= a[key](prevObj);
            var idx= b.template.indexOf('}}', arguments[arguments.length-2])+ 3;
            
            var s2= b.template.substring(0, arguments[arguments.length-2]).split('}}');
            
            x.next= b.template.substring(idx).split('{{')[0];
            x.prev= s2[s2.length-1];
    
            if(typeof(x.num)=='string' && x.num.includes(' *0')) {
                fixQuaNayTruoc(x);
            }
            if(typeof(x.num)=='string' && x.num.includes(' *')) {
                if(DEBUG) console.log('>>> !fixQuaNayTruoc', x);
                if(/(nay|qua|trước)/.exec(x.next)) x.num= x.num.replace(' *', ' -');
                else x.num= x.num.replace(' *', ' +');
            }
    
            b.reg.push(x);
            return ''+ x.tnt;
    
        })
    });
    // console.log(JSON.stringify(b, null, 2));
    // console.log('=>', b.template);

    result.text1= text;
    result.text2= b.template;
    var res= {};
    b.reg.forEach(function(el){
        if(el.num.toString().includes(' ')) {
            // console.log('\t', el.num);
            var kkey= el.num.split(' ');
            if(res[kkey[0]] && !/^[+-]/.exec(res[kkey[0]])) return;
            res[kkey[0]]= kkey[1];
        }
    });
    result.entity= res;
    // console.log(res);
    return result;
}



// zip3= zip3.filter(function(el){
//     if(/{{namtgngay}}/.exec(el)) return true;
//     else return false;
// });

// crack(zip3[Math.floor(Math.random()*zip3.length)])
// crack('ngày {{day}} tháng {{mon}} năm {{yearxx}}');
// crack('một vài {{namtgngay}} tới');
// crack('mùa đông năm {{yearxx}}');
// crack("một vài {{namtgngay}} nay");

// for(var i=0; i<1000;i++) crack(zip3[Math.floor(Math.random()*zip3.length)])

// crack('ngày {{maimot}} {{day}} tháng {{mon}}');
// crack('ngày {{maimot}} tháng {{mon}}')




var str= '';
var checkNeg= require('./isNeg.json');
function sinhDATE(arrTemplate){
    arrTemplate.forEach(function(el, idx){
        // el= 'Album Cánh nâu trong đêm được thực hiện nhờ đâu?';
        if(DEBUG) {
            // el= 'vào {{date}} tôi có vào lúc {{date}} soong soong {{date}}';
            // el= 'cuối {{date}} trước tôi có tham gia ...';
            el= '{{date}} chúng tôi về thăm quê ngoại'
            zip3= [
                // "sau {{numberic}} {{tuanngay}} nữa"
                // 'sau {{numberic}} {{namtgngay}}',
    
                'sau {{numberic}} {{namtgngay}}',
    
                // '{{numberic}} {{namtgngay}} về trước',
                // 'cách đây gần {{mothai}} {{namtgngay}}',
    
                // "sáng ngày {{day}}",
                // "hè năm {{yearxx}}"
    
            ]
    
            console.log('>>>', el);
        }
    
        var result= {};
        var tag= [];
        var prevdelTa= 0;
        // {{date}} len 8
        const TAGG_LEN= 8;
    
        if(!/{{date}}/.exec(el)){
            // console.log(idx, el);
            // if(str.length>1000) {
            //     console.log({str})
            //     process.exit();
            // }
            if(!checkNeg[el]) {
                if(rdx100()> 90){
                    // random <date> data without value in tag
                    result.textdate= '';
                    result.text= el.trim();
                    var arr= el.split(' ');
                    var idxRan= Math.floor(Math.random()*arr.length);
                    arr.slice(0, idxRan).forEach(function(el2){
                        result.textdate+= el2+ ' ';
                    });
                    var endRan= idxRan+ 1+ Math.floor(Math.random()*4);
                    result.textdate+= '<date{}>';
                    arr.slice(idxRan, endRan).forEach(function(el2){
                        result.textdate+= el2+ ' ';
                    });
                    result.textdate= result.textdate.trim();
                    result.textdate+= '</date> ';
                    arr.slice(endRan).forEach(function(el2){
                        result.textdate+= el2+ ' ';
                    });
                    result.textdate= result.textdate.trim();
                } else {
                    result.textdate= el.trim();
                    result.text= el.trim();
                }
                result.tag= [];
                str+= JSON.stringify(result)+ '\n';
                checkNeg[el]= true;
            }
            return;
        }
    
        el= el.replace(/{{date}}/g, function(){
            var idx= arguments[arguments.length-2];
            // console.log(idx);
            var iiiiii= zip3[Math.floor(Math.random()*zip3.length)];
            if(DEBUG) console.log({iiiiii});
            var res= crack(iiiiii);
            tag.push({
                start:      idx+ prevdelTa,
                mid:        idx+ prevdelTa+ res.text2.split(' ')[0].length,
                end:        idx+ prevdelTa+ res.text2.length,
                entity: res.entity
            });
            prevdelTa= prevdelTa-8+ res.text2.length;
            // console.log(res);
            // str2+= JSON.stringify(res.entity);
            return `<date${JSON.stringify(res.entity)}>`+ res.text2+ '</date>';
    
        });
    
        // vào {{date}}
        // đến {{date}}
        // từ {{date}}
        // đầu|cuối|giữa {{date}}
    
        if(el.includes('nữa</date> trước')) return;
        if(el.includes('tới</date> trước')) return;
        if(DEBUG) console.log({el})
    
        if(el.includes('hôm nay')) return;
        if(el.includes('ngày nay')) return;
    
        if(el.includes('nửa tuần')) return;
        if(el.includes('trăm tuần')) return;
        if(el.includes('triệu tuần')) return;
    
        if(/(mươi|mốt|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|chục|mười|kia)(\/|-|\.)[0-9]/.exec(el)) return;
        if(/[0-9](\/|-|\.)(mươi|mốt|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|chục|mười)/.exec(el)) return;
        if(/mùng [1-9][0-9]/.exec(el)) return;
    
        el= el.replace(/([0-9]+) hôm qua/, '$1 ngày trước');
        
        el= el.trim();
        result.textdate= el;
        result.text= require('cheerio').load('<body>'+ el+ '</body>').text().trim();
        result.tag= tag;
        // if(result.tag.length>1) console.log(JSON.stringify(result)+ '\n');
        str+= JSON.stringify(result)+ '\n';
    
        if(DEBUG) {
            console.log(JSON.stringify(result)+ '\n');
            process.exit();
        }
    });
}

str= '';
if(process.argv[2]== 'test'){
    var positive_negative_test= [];
    require('fs').readFileSync(__dirname+ '/positive_negative_test.csv').toString().split(',\r\n').forEach(function(el){
        if(!el) return;
        el= el.replace(/"/g, '').trim();
        if(!el) return;
        positive_negative_test.push(el);
    });
    sinhDATE(positive_negative_test);
    require('fs').appendFileSync(__dirname + '/zip4.test.json', str.toLowerCase());
}
else {
    var positive= [];
    require('fs').readFileSync(__dirname+ '/positive.csv').toString().split(',\r\n').forEach(function(el){
        if(!el) return;
        el= el.replace(/"/g, '').trim();
        if(!el) return;
        positive.push(el);
    });
    sinhDATE(positive);

    var negative= [];
    require('fs').readFileSync(__dirname+ '/negative.csv').toString().split(',\r\n').forEach(function(el){
        if(!el) return;
        el= el.replace(/"/g, '').trim();
        if(!el) return;
        negative.push(el);
    });
    sinhDATE(negative);

    require('fs').appendFileSync(__dirname + '/zip4.res.json', str.toLowerCase());
}

require('fs').writeFileSync(__dirname+ '/isNeg.json', JSON.stringify(checkNeg, null, 1));
