## SangNom2 ##

Reimplementation of the old MarcFD's SangNom filter. Output is not completely but mostly identical.

### What's difference ###
* It's open source
* Correct borders processing
* Additinal colorspace support
* Requries SSE2

Performance right now is mostly identical to the old version despite this plugin using SSE2. Additional improvements are planned.

### Y8 ###
The most important difference for now is Y8 support in AviSynth 2.6. This enables much faster antialiasing (especially when used with [FTurn](https://github.com/tp7/fturn) plugin) without any chroma processing.
```
function maa(clip input, int "mask") {
    mask = input.mt_edge("sobel",7,7,5,5).mt_inflate()
    aa_clip = input.ConvertToY8().Spline36Resize(width(input)*2,height(input)*2).FTurnLeft() \
    			   .SangNom2().FTurnRight().SangNom2().Spline36Resize(width(input),height(input))
    return mt_merge(input,aa_clip,mask,u=2,v=2) 
}
```

### License ###
This project is licensed under the [MIT license](http://opensource.org/licenses/MIT). Binaries are [GPL v2](http://www.gnu.org/licenses/gpl-2.0.html) because if I understand licensing stuff right (please tell me if I don't) they must be.
