
const fs = require('fs');
const config = require('config');
const { promisify } = require('util');
const Promise = require("bluebird");
//const request = require('request-promise-native');
const path = require('path');
const _ = require('lodash');
const numeral = require('numeral');
const execFile = promisify(require('child_process').execFile);

const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

const main = async () => {
    
    const myArgs = process.argv.slice(2);
    
    const outSegments = myArgs[3];
    
    // read segments
    const segments = JSON.parse(await readFile(myArgs[0])).results;

    // read results
    const langClass = JSON.parse(await readFile(myArgs[1] /*"MEDIA_2019_01602888/predict/utt2lang.json"*/));

    // turn lookup each result and add it to the segment into a dictionary
    let lastLang = null;
    let lastDuration = 0;
    let lastStart = 0;
    let lastEnd = 0;
    
    _.each(segments, seg => { 
        // what is the key for lookup?
        const k = ['split', numeral(seg.start/seg.samplerate_num * 1000).format('00000000'), numeral(seg.end/seg.samplerate_num * 1000).format('00000000'), seg.id].join('_');
        
        //console.log(k);
        
        seg.lang = langClass[k];
        //console.log(langClass[k]);
        
        if (seg.lang === lastLang) {
            //lastDuration += (seg.end - seg.start);
            lastEnd = seg.end;
        } else {
            if (lastLang != null) {
                console.log(`Last segment language: ${lastLang} of length: ${lastEnd - lastStart} [${lastStart}, ${lastEnd}]`);
            }
            lastLang = seg.lang;
            //lastDuration = (seg.end - seg.start);
            lastStart = seg.start;
            lastEnd = seg.end;
        }
        
    });
    
    // group segments by language.
    const groupedSegments = _.groupBy(segments, 'lang');
    
    //console.log(JSON.stringify(groupedSegments['en'], '', 4));
    
    // trim WAV segments per language
    
    const extGroupedSegments = {};
    
    await Promise.map( _.keys(groupedSegments), async (langOut) => {
       
        const v = groupedSegments[langOut];
       
        const outWAV = `${myArgs[2]}-${langOut}.wav`;
        // trim using SoX
        const soxArgs = [
            myArgs[2],
            '-c',  '1',
            outWAV,
            'trim'
        ].concat( _.flatten(_.map( v, seg => [`=${seg.start/seg.samplerate_num}`, `=${seg.end/seg.samplerate_num}`] ) ) );
        
        console.log(soxArgs);
        
        const { stdout, stderr } = await execFile('sox', soxArgs);
        
        console.log(stdout);
        
        const ffmpegArgs = [
            '-y',
            '-i',
            `${myArgs[2]}-${langOut}.wav`,
            `${myArgs[2]}-${langOut}.mp3`
        ]

        const { stdout: stdout1, stderr: stderr1 } = await execFile('ffmpeg', ffmpegArgs);
        
        console.log(stdout1);    
       
        extGroupedSegments[langOut] = {
            segments: v,
            wav: path.resolve(outWAV) // resolve to an absolute path to enable easier pass-on to other apps.
        };
        
    }, {concurrency: 1});
    
   // consolidate
   //await writeFile()
   
   await writeFile(outSegments, JSON.stringify(extGroupedSegments, '', 2));
  
   
}

main();

