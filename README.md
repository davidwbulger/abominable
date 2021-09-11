# abominable
Python library for converting drum tablature to audio.

This repo contains the "Abominable" package for drum synthesis and tablature-to-audio conversion, and a few demos showing how to use it.

## Project status

This project is very new. If it's of interest to anyone else, I'm planning to turn it into an open-source project. I'm open to hear your suggestions for improvements, whether or not you're a programmer.

## How to use Abominable

**Note that currently you need to have a Python installation available in order to use Abominable, and you may need to install some additional modules.**

The two main functionalities that Abominable offers are

* converting drum tablature to audio, and
* synthesising drum sounds.

These are independent, in the sense that if you'd prefer to use sampled drum-hits, you can do that too. In fact, there are three kinds of drum sounds you can use:

* pre-sampled audio,
* built-in standard drums ("tom", "crash" et cetera),
* additional, user-created drum sounds (created by writing a small Python function).

Take a look at the three demo files to see how Abominable is used:

### `demo1.txt`

This short example shows how to create a well-known rhythm using the built-in hihat, snare and bass drum. Open the file in a text editor to see its contents. To run it, type `python Abominable.py demo1.txt` at a system command line. (You must have a Python system installed in order to use Abominable.)

### `demo2.txt`

This slightly more involved example also shows how you can define sections, and then combine them into larger sections until eventually you've defined a whole song's drum track. Again, to run it, type `python Abominable.py demo2.txt` at a system command line. [You can hear the drums in the context of the song on YouTube.](https://www.youtube.com/watch?v=pq9Y4TQswVU)

### `demo3.py`

This example demonstrates how to extend Abominable's built-in drum kit by defining a new drum of your own. (This is considered advanced usage; most users might never want to do it.) In order to do that, you need to write a little bit of Python code, and therefore, this demo works a little differently: note that the file has a `.py` extension, and you run it by typing `python demo3.py` at the system command prompt.

As illustrated in `demo3.py`, you create a new drum by defining a function that returns a Python dictionary, and then you make it available by adding it to Abominable's drum collection in `drum\_dict`. The dictionary that your drum function returns should specify some or all of the parameters of Abominable's quite general function `note` (search for `def note` in `Abominable.py` to see these parameters).

## Synthesis quality

As of the initial commit, the quality of the drum sythesis is fairly low. Some of the drums sound more believable than others. You might decide that the drums sound too fake for your purposes. I'm open to ideas about how to achieve more realism, but in the interim I'd like to explain broadly why they're structured the way they are.

Firstly, I'm just one guy, with no real background in audio synthesis, so I figured competing with professional software on that front was futile. In lieu of that, the next most obvious thing to do would be to base the synthesis code on the "basics of drum synthesis" as they're usually presented (an ADSR envelope applied to stationary noise, and that kind of thing). I deliberately have not quite done that, because that would simply reproduce 1980s/Casiotone-style drums, and I figure people who want that sound can already get it elsewhere. That is, I've chosen fairly simple methods that will produce a sound palette that's admitedly a bit fake-sounding, but at least fake in a unique way.

## Directions

One feature I'm considering adding is a MusicXML export option. That would allow you to convert drum tablature to MusicXML, which you could then import into MuseScore or Finale or something like that, which in turn would produce a better audio export. Personally I find drum entry particularly drudgerous in MuseScore and Finale, so it seems to me that it would be a useful feature. I haven't given the design much thought though.

An obvious accessibility improvement would be to release a binary version, so that users wouldn't need their own Python installations. As well as the technical issues, I'm a little wary of the legalities, since the dependencies have their own various licenses.

## Comment or contribute

Probably the surest way to reach me is by email at [my github name]@gmail.com. I'll be delighted to hear from anyone with questions, suggestions, et cetera.
