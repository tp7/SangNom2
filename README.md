## SangNom2 ##

Reimplementation of the old MarcFD's SangNom filter. Output is not completely but mostly identical.

### What's different ###
* It's open source
* Correct borders processing
* Additinal colorspace support
* Multithreading support
* Requries SSE2

Singlethreaded performance is mostly identical to the old version despite this plugin using SSE2.

### Y8 ###
One of the most important differences is Y8 support in AviSynth 2.6. This enables much faster antialiasing (especially when used with [FTurn](https://github.com/tp7/fturn) plugin) without any chroma processing.
```
function maa(clip input) {
    mask = input.mt_edge("sobel",7,7,5,5).mt_inflate()
    aa_clip = input.ConvertToY8().Spline36Resize(width(input)*2,height(input)*2).FTurnLeft() \
    			   .SangNom2().FTurnRight().SangNom2().Spline36Resize(width(input),height(input))
    return mt_merge(input,aa_clip,mask,u=2,v=2) 
}
```
### Multithreading ###
This plugin uses min(number of logical processors, 4) threads to do its job. You can control number of threads using the *threads* parameter. In my tests performance doesn't get any better when using more than 4 threads.

Internally it uses a simple thread pool but I do consider switching to avstp if it gets a bit nicer api.

### Chroma processing ###
Originally, SangNom always assumes aa=0 for chroma processing. This makes some use cases harder to implement, so additional parameter *aac* was introduced. It's the same as *aa* but for chroma. Default value is 0 to maintain backward compatibility.

### License ###
This project is licensed under the [MIT license](http://opensource.org/licenses/MIT). Binaries are [GPL v2](http://www.gnu.org/licenses/gpl-2.0.html) because if I understand licensing stuff right (please tell me if I don't) they must be.
