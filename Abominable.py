# ABOMINABLE: free, open-source utility to convert drum tablature to audio.

import numpy as np
import os
import re
import scipy.io.wavfile as wav
import simpleaudio as sa
from sly import Lexer, Parser
import sys
import types
import warnings

rng = np.random.default_rng(seed=13)

##  CLASSES  ##################################################################

class Drum:
    # Each object of this class represents a description of a sound (that is, a
    # tablature line), but is responsible for creating that sound at various
    # tempos.
    rvol = re.compile("^vol[+-]")

    def __init__(self, input, sound_folder, fs):
        self.sound_folder = sound_folder
        self.fs = fs
        self.vol = 0  #  dB, relative to other drums
        input = input.split()
        self.brush = "brush" in input
        if self.brush: input.remove("brush")
        for k in range(len(input)-1,-1,-1):
           if Drum.rvol.match(input[k]):
               try:
                   dvol = float(input[k][3:])
                   self.vol += dvol
                   input.pop(k)
               except:
                   pass
        if len(input)<1:
            raise ValueError("Too little info to specify drum sound")
        self.sampled = (input[0]=="soundfile")
        if self.sampled:
            if len(input) != 2:
                raise ValueError("soundfile syntax is:\n" +
                    "soundfile FILENAME [brush] [vol+/-N]")
            self.file_name = input[1]
            self.abs_dur = True
            self.audio = None  #  to be loaded when/if required
        else:
            self.instrument_name = input.pop(0)
            self.note_func = drum_dict.get(self.instrument_name.lower())
            if (self.note_func is None) or \
                not isinstance(self.note_func,types.FunctionType):
                raise ValueError("No code found to generate instrument " +
                    self.instrument_name.lower())
            if len(input)<1:
                raise ValueError("Duration (absolute or relative) is needed "+
                    "when specifying a generated sound.")
            try:
                self.dur = float(input[0])
                self.abs_dur = False
                self.audio = {}  #  To be loaded or generated as required
            except:
                if input[0][-1] in "sS":
                    try:
                        self.dur = float(input[0][:-1])
                        self.abs_dur = True
                        self.audio = []  #  to be filled as required
                    except:
                        raise ValueError("Cannot interpret duration "+input[0])
                else:
                    raise ValueError("Cannot interpret duration "+input[0])
            input.pop(0)
            self.label = self.instrument_name.lower() + "_" + "_".join(input) \
                + ("_brush" if self.brush else "")
            # Convert parameters that look like numbers to float type:
            for k in range(len(input)):
                try:
                    input[k] = float(input[k])
                except:
                    pass

            # The instrument function, e.g., "abominable_kazoo", takes usually
            # just one or two input parameters, but processes them somehow to
            # produce parameters for the much more general function "note". The
            # parameters are stored in the dictionary param_dict, and then we
            # explicitly add the brush and sample freq parameters.
            self.param_dict = self.note_func(*input)
            self.param_dict.update({'brush':self.brush,'fs':self.fs})

    def get_audio(self,tempo):
        # Returns the audio for this note at this tempo, in one of three ways:
        # it loads it from disk, if it's just a sampled sound, or if it's a
        # generated sound that we've used before & still have on file.
        # Otherwise it generates it algorithmically.
        # For all three options, the audio will be cached in memory while this
        # Drum object persists.
        if((self.audio is None) if self.abs_dur else(tempo not in self.audio)):
            cur_dur = self.dur
            if not self.abs_dur:
                cur_dur *= 60/tempo
            path = self.path(cur_dur)
            wavdat = self.read_audio_from_file(path)
            if wavdat is None:
                if self.sampled:
                    raise ValueError("Unable to load sound" + self.file_name)
                print("Generating sound " + path)
                wavdat = note(dur=cur_dur,**self.param_dict)
                self.write_audio_to_file(path,wavdat)
            # Now store it in self.audio for faster retrieval next time:
            if self.abs_dur:
                self.audio = wavdat
            else:
                self.audio[tempo] = wavdat
        return (self.audio if self.abs_dur else self.audio[tempo])

    def get_audio(self,tempo,timbix):
        # Returns the audio for this note at this tempo, in one of three ways:
        # it loads it from disk, if it's just a sampled sound, or if it's a
        # generated sound that we've used before & still have on file.
        # Otherwise it generates it algorithmically.
        # For all three options, the audio will be cached in memory while this
        # Drum object persists.
        # If it's generated, timbix ("timbre index") says which IID version of
        # the note to use.
        if self.sampled:
            if self.audio is None:
                path = self.path()
                self.audio = self.read_audio_from_file(path)
                if self.audio is None:
                    raise ValueError("Unable to load sound" + self.file_name)
            return self.audio
        # Otherwise, it's generated:
        if not (self.abs_dur or tempo in self.audio):
            self.audio[tempo] = []
        notelist = self.audio if self.abs_dur else self.audio[tempo]
        if timbix >= len(notelist):
            timbix = len(notelist)
            cur_dur = self.dur
            if not self.abs_dur:
                cur_dur *= 60/tempo
            path = self.path(cur_dur,timbix)
            wavdat = self.read_audio_from_file(path)
            if wavdat is None:
                print(f"Generating version {timbix} of sound " + path)
                wavdat = note(dur=cur_dur,**self.param_dict)
                self.write_audio_to_file(path,wavdat)
            # Now store it in self.audio for faster retrieval next time:
            notelist.append(wavdat)
        return notelist[timbix]

    def path(self,duration=0,timbix=None):
        fname = self.file_name if self.sampled else \
            self.label+f"_dur{duration:.3f}"
        if timbix is not None:
            fname += f"_v{timbix}"
        return os.path.join(self.sound_folder, fname + ".wav")

    def read_audio_from_file(self,path):
        # return None if not found; warn but load Lchan if stereo; fail if fs
        if os.path.isfile(path):
            print("Loading " + ("sampled" if self.sampled else "pre-generated")
                + " sound " + path)
            try:
                (fs,wavdat) = wav.read(path)
            except:
                raise ValueError(f"File {path} found, but unreadable as wav.")
            if fs != self.fs:
                raise ValueError("All sample rates must be equal.")
            if wavdat.ndim>1:
                warnings.warn(f"File {path} has multiple channels (for " +
                    "instance, stereo). Only the first channel will be used.")
                wavdat = wavdat.reshape((wavdat.shape[0],-1))[:,0]
            return wavdat
        else:
            return None

    def write_audio_to_file(self,path,wavdat):
        # This stores the sound of a single drum for later re-use. (See also
        # Sequence.export_wav, which writes the whole tune as audio.)
        wav.write(path, int(self.fs), wavdat)

# The next two objects are the two types of thing that are stored in a sequence
# in the Sequence object. The idea is that a tune or part thereof is an
# initial TempoSetting, followed by a number of TabSections with, possibly,
# additional TempoSettings interspersed (to speed up or slow down the tempo).

class TempoSetting:
    def __init__(self, value, absolute):
        self.value=value  #  either a tempo, or a change in tempo
        self.absolute=absolute  #  if false, then value is an in-(de-)crement

class TabSection:
    def __init__(self, gridlist):
        if min(len(grid) for grid in gridlist) < \
            max(len(grid) for grid in gridlist):
            raise ValueError("All tablature snippets must have the same " +
                "number of lines.")
        self.grid = ["".join(line) for line in zip(*gridlist)]
        if np.any(np.diff([len(line) for line in self.grid])!=0):
            raise ValueError("All tablature lines must be of equal length.")
        if len(gridlist)<2:  #  otherwise it's already been checked
            N = len(gridlist)
            if any(gridlist[j][k]=='|' and gridlist[jplus1][k]!='|'
                for k in range(len(gridlist[0]))
                for (j,jplus1) in zip(range(N),list(range(1,N))+[0])):
                raise ValueError("Barline positions must agree in all lines.")

class Sequence:
    def __init__(self, seqels):
        # Input arg is a sequence of TempoSettings and TabSections. Any
        # adjacent TabSections can be concatenated; otherwise, we leave it as
        # is for now. (Absolute tempos, timings and durations will be computed
        # in the audio export routine.)
        for k in range(len(seqels)-1,0,-1):
            if isinstance(seqels[k-1],TabSection) and \
                isinstance(seqels[k],TabSection):
                seqels[k-1] = TabSection([seqels[k-1].grid,seqels.pop(k).grid])
        self.seqels = seqels  #  sequence elements

    def len(self):
        return sum(len(seqel.grid[0]) for seqel in self.seqels if
            isinstance(seqel,TabSection))

    def export_txt(self,file_name,col_count):
        # Remove tempo info:
        grid = Sequence([seqel for seqel in self.seqels if
            isinstance(seqel,TabSection)]).seqels[0].grid
        J = len(grid)     #  number of tab rows
        K = len(grid[0])  #  number of tab columns
        col_count = int(col_count)
        numblocks = 1 + (K-1)//col_count
        with open(file_name, 'w') as fid:
            for k in range(numblocks-1):
                for j in range(J):
                    fid.write(grid[j][col_count*k:col_count*(k+1)]+"\n")
                fid.write("\n\n")  #  gap between blocks
            for j in range(J):
                fid.write(grid[j][col_count*(numblocks-1):]+"\n")

    def export_wav(self, file_name, the_parser):
        self.export_wav_or_mid(file_name, the_parser, "wav")

    def export_mid(self, file_name, the_parser):
        self.export_wav_or_mid(file_name, the_parser, "mid")

    def export_wav_or_mid(self, file_name, the_parser, wav_or_mid):
        # Note, we're passing in a reference to the parser too, since it knows
        # the instrument definitions, the sample rate, et cetera.
        drums = [Drum(input, the_parser.sound_folder, the_parser.fs)
            for input in the_parser.tab_lines]
        if any(len(tabsec.grid) != len(drums) for tabsec in self.seqels if
            isinstance(tabsec, TabSection)):
            raise ValueError("All tab sections must have exactly one line " +
                "for each instrument defined by tablines.")
        if len(self.seqels)<1:
            raise ValueError("Trying to export empty sequence.")
        if not(isinstance(self.seqels[0],TempoSetting) and
            self.seqels[0].absolute):
            if the_parser.tempo is None:
                tempo = 144
                warnings.warn(
                    "No initial tempo found; default 144 columns/minute used.")
            else:
                tempo = the_parser.tempo

        # First, build a list of all notes required, with start times.
        note_list = []
        time = 0
        for seqel in self.seqels:
            if isinstance(seqel, TempoSetting):
                if seqel.absolute:
                    tempo = seqel.value
                else:
                    tempo += seqel.value
            else:  #  seqel must be a TabSection
                # We know that the lines have the same lengths & barline
                # positions, so we'll use the first line as a template.
                offset = 0  #  number of beats since start of TabSection
                for (p,col) in enumerate(zip(*seqel.grid)):
                    if col[0] != "|":
                        for (j,glyph) in enumerate(col):
                            if glyph>='1' and glyph<= '9':
                                # Store instrument index & nominal vol & time:
                                note_list.append((j, int(glyph), tempo, time +
                                    offset*60/tempo + the_parser.rand_time *
                                    rng.standard_normal()))
                        offset += 1
                time += offset*60/tempo

        if wav_or_mid == "wav":
            # Start with a zero vector and add in each sound. We'll actually
            # work backwards through the list we've just built, as a heuristic,
            # because it's not easy to know in advance exactly how long the
            # sound vector needs to be. This way, we can resize as necessary to
            # fit each note, but will probably only need to do it a few times.
            wavmix = np.array([])  #  0-length, 1-D vector of float64s
            for (instr, nomvol, tempo, time) in reversed(note_list):
                if time<0: time=0
                startsamp = int(time*the_parser.fs)
                wavdat = drums[instr].get_audio(tempo,
                    rng.integers(the_parser.rand_timbre))
                if (overshoot := startsamp+len(wavdat)-len(wavmix)) > 0:
                    wavmix = np.concatenate((wavmix, np.zeros(overshoot)))
                finalvol = 2*nomvol + drums[instr].vol + \
                    the_parser.rand_vol*rng.standard_normal()
                wavmix[startsamp:startsamp+len(wavdat)] += \
                    wavdat * np.power(10, 0.1*finalvol)
            wav.write(file_name, int(the_parser.fs),
                (wavmix*32767/np.max(np.abs(wavmix))).astype(np.int16))
        else:  #  export MIDI instead
            # According to
            # https://www.hedsound.com/p/midi-velocity-db-dynamics-db-and.html,
            # volume in dB is equal to 40*log10(MIDIVelocity/127).

            # Here, the volume digit d (1--9) maps to a MIDI "velocity" of
            # 8--120, via 14*d-6. These can then be modified by random
            # variation and vol modifiers. Anything ending up outside the range
            # 1--127 is clipped to that range. For simplicity, each 'decibel'
            # of modification corresponds to a velocity change of 2 (though
            # that doesn't strictly agree with the wav output, assuming MIDI is
            # implemented according to the hedsound info).

            # Use drums[instr].dur, drums[instr].abs_dur and tempo to calculate
            # note duration.
            notes = [(drums[j].param_dict.get('midi',41),
                max(1,min(127,int(14*v+7*(drums[j].vol+
                the_parser.rand_vol*rng.standard_normal())-6))), s,
                drums[j].dur*(1 if drums[j].abs_dur else 60/t))
                for (j,v,t,s) in note_list]
            write_smf0(notes, file_name)

class AbomLexer(Lexer):
    tokens = {TABLINES, TAB, SEQUENCE, REPEAT, RELATIVETEMPO, NUMBER, LPAR,
        RPAR, SOUNDFOLDER, SAMPFREQ, RANDVOL, RANDTIME, RANDTIMBRE, TEMPO,
        EXPORT,PATHORID}

    ignore = ' \t\n'
    ignore_comment = r'\#.*'

    # "Supertokens" for Tablines, Tab and Sequence:
    # These are handled separately because otherwise I don't know how to apply
    # separate lexing rules to the (potentially multiple) lines themselves.

    @_(r'tablines[ \t=]*(#.*)?\n(([ \t]*[a-z].*\n)+)[ \t]*(#.*)?\n')
    def TABLINES(self, t):
        match=re.search(r'tablines[ \t=]*(#.*)?\n(([ \t]*[a-z].*\n)+)[ \t]*' +
            r'(#.*)?\n', t.value)
        t.value = [re.sub(r"^[ \t]*([^#]*[^# \t])[ \t]*(#.*)?$",
            lambda m: m.group(1), line) for line in
            match.group(2).split('\n')[:-1]]
        return t

    @_(r'tab[ \t]+([a-zA-Z][a-zA-Z0-9_]*)[ \t=]*(#.*)?\n' + \
            r'(([ \t]*[-1-9|]+[ \t]*(\#.*)?\n)+)[ \t]*(#.*)?\n')
    def TAB(self, t):
        match = re.search(r'tab[ \t]+([a-zA-Z][a-zA-Z0-9_]*)[ \t=]*(#.*)?\n'+ \
            r'(([ \t]*[-1-9|]+[ \t]*(\#.*)?\n)+)', t.value)
            #r'(([ \t]*[-1-9|]+[ \t]*(\#.*)?\n)+)[ \t]*(#.*)?\n', t.value)
        t.value = (match.group(1), [re.sub(r"^[ \t]*([-1-9|]+)[ \t]*(\#.*)?$",
            lambda m: m.group(1), line)
            for line in match.group(3).split('\n')[:-1]])
        return t

    @_(r'sequence[ \t]+([a-zA-Z][a-zA-Z0-9_]*)[ \t=]*')
    def SEQUENCE(self, t):
        t.value = re.search(r'sequence[ \t]+([a-zA-Z][a-zA-Z0-9_]*)[ \t=]*',
            t.value).group(1)
        return t

    # Nonkeyword tokens:

    @_(r'[xX*]\d+')
    def REPEAT(self, t):
        t.value = int(t.value[1:])
        return t

    @_(r'([\+\-])[ \t]*([0-9]*([0-9]\.?|\.[0-9])[0-9]*)')
    def RELATIVETEMPO(self, t):
        match = re.search(r'([\+\-])[ \t]*([0-9]*([0-9]\.?|\.[0-9])[0-9]*)',
            t.value)
        t.value = float(match.group(2)) * (1 if match.group(1)=="+" else -1)
        return t

    @_(r'[0-9]*([0-9]\.?|\.[0-9])[0-9]*')
    def NUMBER(self, t):
        t.value = float(t.value)
        return t

    LPAR = r'[\(\[\{]'
    RPAR = r'[\)\]\}]'

    # Keywords optionally followed DIRECTLY by an "=":
    SOUNDFOLDER = r'soundfolder[ \t=]*'
    SAMPFREQ = r'sampfreq[ \t=]*'
    RANDVOL = r'randvol[ \t=]*'
    RANDTIME = r'randtime[ \t=]*'
    RANDTIMBRE = r'randtimbre[ \t=]*'
    TEMPO = r'tempo[ \t=]*'

    # Other keywords:
    EXPORT = 'export'

    # filepath-or-tabsequenceID, defined last, to only match if the keywords
    # don't:
    PATHORID = r'[.a-zA-Z_\\\/][.a-zA-Z_\\\/0-9]*'

class AbomParser(Parser):
    tokens = AbomLexer.tokens
    # debugfile = 'pardump.txt'

    def __init__(self):
        self.seq_dict = {}  #  dictionary of sections & their names
        self.sound_folder = "./Sounds"
        self.fs = 44100.0  #  sampling frequency in Hz
        self.rand_vol = 0
        self.rand_time = 0
        self.rand_timbre = 4
        self.tab_lines = []
        self.tempo = None

    @_('SOUNDFOLDER PATHORID')
    def statement(self, p):
        self.sound_folder = p.PATHORID
        print(f"Setting soundfolder to '{p.PATHORID}'.")

    @_('SAMPFREQ NUMBER')
    def statement(self, p):
        print(f"Setting sampfreq (the sample frequency, in Hz) to {p.NUMBER}.")
        self.fs = p.NUMBER

    @_('RANDVOL NUMBER')
    def statement(self, p):
        print("Setting randvol (the standard deviation of notes' loudness, " +
            f"in dB) to {p.NUMBER}.")
        self.rand_vol = p.NUMBER

    @_('RANDTIME NUMBER')
    def statement(self, p):
        print("Setting randtime (the standard deviation of notes' timings, " +
            f"in seconds) to {p.NUMBER}.")
        self.rand_time = p.NUMBER

    @_('RANDTIMBRE NUMBER')
    def statement(self, p):
        print("Setting randtimbre (how many versions of each note to " +
        f"generate) to {p.NUMBER}.")
        self.rand_timbre = p.NUMBER

    @_('TABLINES')
    def statement(self, p):
        self.tab_lines = p.TABLINES

    @_('TAB')
    def statement(self, p):
        self.seq_dict[p.TAB[0]] = Sequence([TabSection([p.TAB[1]])])
        print(f"Defining sequence '{p.TAB[0]}', of length {len(p.TAB[1][0])}.")
        # Or equivalently:
        # print(f"Defining sequence '{p.TAB[0]}', of length " +
        #     f"{self.seq_dict[p.TAB[0]].len()}.")

    @_('EXPORT PATHORID PATHORID NUMBER')
    def statement(self, p):
        if p.PATHORID0 in self.seq_dict and len(p.PATHORID1)>4 and \
            p.PATHORID1[-4:]==".txt" and p.NUMBER>=1:
            self.seq_dict[p.PATHORID0].export_txt(p.PATHORID1,p.NUMBER)
            print(f"Exporting compiled tablature to file '{p.PATHORID1}'.")
        else:
            raise ValueError("The export command has three valid uses:\n" +
                "export PREVIOUSLY_DEFINED_SEQUENCE FILE_NAME.txt " +
                "POSITIVE_COLUMN_COUNT\nexport PREVIOUSLY_DEFINED_SEQUENCE " +
                "FILE_NAME.wav\nexport PREVIOUSLY_DEFINED_SEQUENCE " +
                "FILE_NAME.mid")

    @_('EXPORT PATHORID PATHORID')
    def statement(self, p):
        if p.PATHORID0 in self.seq_dict and len(p.PATHORID1)>4 and \
            p.PATHORID1[-4:] in {".wav", ".mid"}:
            if p.PATHORID1[-3:]=="wav":
                print("Generating audio...")
                self.seq_dict[p.PATHORID0].export_wav(p.PATHORID1,self)
                print(f"Audio exported to file '{p.PATHORID1}'.")
            else:
                self.seq_dict[p.PATHORID0].export_mid(p.PATHORID1,self)
                print(f"MIDI exported to file '{p.PATHORID1}'.")
        else:
            raise ValueError("The export command has three valid uses:\n" +
                "export PREVIOUSLY_DEFINED_SEQEUNCE FILE_NAME.txt " +
                "POSITIVE_COLUMN_COUNT\nexport PREVIOUSLY_DEFINED_SEQEUNCE " +
                "FILE_NAME.wav\nexport PREVIOUSLY_DEFINED_SEQEUNCE " +
                "FILE_NAME.mid")

    @_('statement statement')
    def statement(self, p):
        pass  #  Allows processing of multiple consecutive statements.

    # Next we handle sequence assignment. This is a little hacky, because the
    # rules will treat every PATHORID as a sequence id. However, if all other
    # valid rules including PATHORID operands appear prior to this stuff, it
    # should be okay.
    @_('PATHORID')
    def seq_spec(self, p):
        return self.seq_dict[p.PATHORID]

    @_('TEMPO NUMBER')
    def seq_spec(self, p):
        return Sequence([TempoSetting(p.NUMBER,True)])

    @_('TEMPO RELATIVETEMPO')
    def seq_spec(self, p):
        return Sequence([TempoSetting(p.RELATIVETEMPO,False)])

    @_('seq_spec REPEAT')
    def seq_spec(self, p):
        return Sequence(p.seq_spec.seqels * p.REPEAT)

    @_('seq_spec seq_spec')
    def seq_spec(self, p):
        return Sequence(p.seq_spec0.seqels + p.seq_spec1.seqels)

    @_('LPAR seq_spec RPAR')
    def seq_spec(self, p):
        return p.seq_spec

    @_('SEQUENCE seq_spec')
    def statement(self, p):
        self.seq_dict[p.SEQUENCE] = p.seq_spec
        print(f"Defining sequence '{p.SEQUENCE}', of length " +
            f"{self.seq_dict[p.SEQUENCE].len()}.")

    @_('seq_spec')
    def statement(self, p):
        if len(p.seq_spec.seqels)==1 and \
            isinstance(p.seq_spec.seqels[0],TempoSetting) and \
            p.seq_spec.seqels[0].absolute:
            self.tempo = p.seq_spec.seqels[0].value

##  NON-CLASS FUNCTIONS  ######################################################

def freqIt(freqFun):
    # An iterator to yield all frequencies below 20kHz
    f = freqFun(k:=0)
    while f < 20000:
        yield f
        k+=1
        f = freqFun(k)

def freqs_metallic(fundFr):
    # Linear overtone sequence
    return lambda k: fundFr*(1+k)

def freqs_circular(fundFr):
    # Simple approx to otone sequence for circular membrane
    return lambda k: fundFr*(1+2*k/3)

def overtone(mu,  #  frequency at centre of spike
    sigma,        #  Gaussian widths of spikes
    L,            #  length of sample
    fs):          #  sampling frequency
    Laug=2**int(1+np.log(L-1)/np.log(2)) # L rounded up if nec to be a pow of 2
    fr = np.linspace(0,fs/2,Laug//2+1)
    spec = np.exp(-(fr-mu)**2/(2*sigma**2))/np.sqrt(sigma)
    randPhase = np.concatenate(([1],
        np.exp(2j*np.pi*np.random.rand(Laug//2-1)), [1]))
    spec = randPhase * spec.astype(complex)
    return np.fft.irfft(spec)[:L]

def ab_env(a, # Attack duration. Will ramp exponentially from -50dB to 0dB.
    b,        # Decay rate in dB/s. Will fade exponentially.
    c,        # Peak amplitude.
    open,     # exponent for extra power-law factor in attack (mainly for wash)
    close,    # exponent for extra power-law factor in decay (mainly for snare)
    L,        # length of envelope
    fs):      # sample rate in Hz
    t = np.arange(L)/fs
    if a<=0:
        return c*np.power(0.1,0.1*b*t)*np.power(1-t/t[-1],close)
    else:
        dblev = np.where(t<a, -50*(t/a-1)**2,
            np.where(t<2*a, -0.5*(t-a)**2*b/a, (1.5*a-t)*b))
        return c*np.power(10.0,0.1*dblev)*np.power(1-t/t[-1],
            close)*np.power(t/t[-1],open)

def note(dur,    #  note duration in seconds
    freqs,       #  func mapping k to kth otone's frequency in Hz
    sharpness,   #  dB diff between peak & trough between overtones
    a0=0, ap=0,  #  kth otone's attack duration will be a0+k*ap seconds
    decay=lambda k:0, #  func mapping k to kth otone's env's decay rate in dB/s
    peak=lambda k:1,  #  func mapping k to kth otone's peak amp
    open=lambda k:0,  #  func mapping k to exponent for extra power-law attack
    close=lambda k:0.01,  #  similar to 'open' but for decay
    brush=False, #  whether to hit with a brush instead of a stick
    fs=44100,    #  sampling frequency in Hz
    **kwargs):   #  in case there are any surplus args (e.g., midi)
    L = int(dur*fs)  #  length of required sample
    sigmaFactor = np.sqrt(1.25/(np.max((sharpness,0.1))*np.log(10)))
    wave = sum((
        ab_env(a0+k*ap,decay(k),peak(k),open(k),close(k),L,fs) *
        overtone(mu,mu*sigmaFactor/(k+1),L,fs)
        for (k,mu) in enumerate(freqIt(freqs))))
    if brush:
        # Pinken the sound:
        wave[1:] += wave[:-1]
        # Sound of one fibre touching surface:
        tick=note(0.03, freqs_circular(1200), 5, 0.0001, 0, lambda k:800*(k+1),
            lambda k:1, lambda k:0, lambda k:0.01, False,fs).astype(np.float64)
        tickSamps = np.insert(1+np.sort(rng.choice(int(0.09*fs),29,False)),0,0)
        tickSamps = tickSamps[tickSamps<len(wave)-len(tick)]
        wave *= np.concatenate([(k+1)*np.ones(d)
            for (k,d) in enumerate(np.diff(np.append(tickSamps,L)))])
        tick *= 0.1/32767 * np.max(np.abs(wave))
        for k in tickSamps:
            if k+len(tick) <= L:
                wave[k:k+len(tick)] += tick * (1+rng.random())
    return (wave*32767/np.max(np.abs(wave))).astype(np.int16)

def abominable_crash(fund_freq=230):
    return {'freqs':lambda k:fund_freq*(1+k/3),
        'sharpness':8,
        'ap':1e-5,
        'decay':lambda k:7+0.003*k,
        'peak':lambda k: rng.random()**3*(1+2.5/(k+1)),
        'midi':49}

def abominable_wash(fund_freq=230):
    return {'freqs':lambda k:fund_freq*(1+k/3),
        'sharpness':12,
        'peak':lambda k:rng.random()**3*(1+2.5/(k+1)),
        'open':lambda k:2,
        'close':lambda k:2}

def abominable_tambourine(fund_freq=4500,sharpness=12):
    return {'freqs':lambda k: fund_freq + (50*k if k<220 else np.inf),
        'sharpness':sharpness,
        'a0':0.02,
        'peak':lambda k:rng.random()**3*np.exp(-(np.abs(k-110)-60)**2/2000),
        'midi':54}

def abominable_ride(fund_freq=50):
    return {'freqs':freqs_metallic(fund_freq),
        'sharpness':8,
        'ap':1e-5,
        'decay':lambda k:5+0.01*k,
        'peak':lambda k:rng.random()**3*(1-10/(k+20)),
        'midi':51}

def abominable_hihat(clopen="closed",fund_freq=3600):
    return {'freqs':lambda k:fund_freq*(1+0.04*k),
        'sharpness':40,
        'ap':1e-5,
        'decay':lambda k:8,
        'peak':lambda k:rng.random()**2,
        'close':((lambda k:0.01) if clopen=="open" else
        (lambda k:0.05+k*0.016)),
        'midi':42}

def abominable_snare(fund_freq=180):
    return {'freqs':freqs_circular(fund_freq),
        'sharpness':3,
        'ap':0.00015,
        'decay':lambda k:np.interp(k,
        [0,5,7,28,32,40,48],[60,80,100,100,75,75,100]),
        'peak':lambda k:rng.random()**2+5/(k+1)**2,
        'midi':38}

def abominable_tom(fund_freq=180):
    return {'freqs':freqs_circular(fund_freq),
        'sharpness':8,
        'a0':0.002,
        'decay':lambda k:np.min((90+0.02*k, 22+4*k)),
        'peak':lambda k:0.96**k*np.exp(0.5*np.cos(2*np.pi*(k-1)/7)),
        'midi':47}

def abominable_plane(fund_freq=180, sharpness=8):
    return {'freqs':freqs_circular(fund_freq),
        'sharpness':sharpness,
        'a0':0.002,
        'decay':lambda k:np.min((90+0.02*k, 22+4*k)),
        'peak':lambda k:0.96**k*np.exp(0.5*np.cos(2*np.pi*(k-1)/7))}

def abominable_bass(fund_freq=20):
    return {'freqs':freqs_circular(fund_freq),
        'sharpness':5,
        'decay':lambda k:np.min((375, 24+8*k)),
        'peak':lambda k:0.98**k,
        'midi':35}

def abominable_sticks(fund_freq=100):
    return {'freqs':lambda k:fund_freq*((k+1)+3/(k+10)*rng.standard_normal()),
        'sharpness':5,
        'decay':lambda k:110-24*np.cos(2*np.pi*k/10),
        'peak':lambda k:rng.random()**0.3*k/(240+k),
        'midi':31}

def abominable_clave(fund_freq=1220):
    return {'freqs':freqs_metallic(fund_freq),
        'sharpness':25,
        'decay':lambda k:200,
        'peak':lambda k:rng.random()+(12 if k<1 else 0),
        'midi':75}

def abominable_fibre():
    return {'freqs':freqs_circular(1200),
        'sharpness':5,
        'a0':0.0001,
        'decay':lambda k:800*(k+1)}

def abominable_triangle(fund_freq=460):
    return {'freqs':freqs_circular(fund_freq),
        'sharpness':3e9,
        'decay':lambda k:5+0.1*k,
        'peak':lambda k:np.exp(-0.001*(k-12)**2),
        'midi':81}

def abominable_cowbell(fund_freq=700):
    return {'freqs':lambda k:fundfreq*(1+0.25*k),
        'sharpness':170,
        'decay':lambda k:100+k,
        'peak':lambda k:rng.random()*(
        12 if k<1 else 4 if k in [4,8] else 2 if k%4==0 else 1),
        'midi':56}

def abominable_midi(code):
    # This should only be used for MIDI export, so the freqs and sharpness
    # parameters are immaterial, but we need to supply them, since they're the
    # note parameters that have no defaults.
    return {'freqs':freqs_circular(180),
        'sharpness':8,
        'midi':code}

drum_dict = {
  "crash":abominable_crash,
  "wash":abominable_wash,
  "tambourine":abominable_tambourine,
  "ride":abominable_ride,
  "hihat":abominable_hihat,
  "snare":abominable_snare,
  "tom":abominable_tom,
  "plane":abominable_plane,
  "bass":abominable_bass,
  "sticks":abominable_sticks,
  "clave":abominable_clave,
  "fibre":abominable_fibre,
  "triangle":abominable_triangle,
  "cowbell":abominable_cowbell,
  "midi":abominable_midi}

##  MIDI DRUM TRACK OUTPUT FUNCTIONS  #########################################

def write_smf0(notes, filename):
    # Write a "Standard MIDI File" of "Type 0."
    # Each note in notes is a (pitch,volume,start,duration) tuple.

    # Set tempo to default to 500000 microseconds per 'crotchet':
    databytes = [0, 255, 81, 3] + encode_int(500000,3)

    # Set up the list of note-on and note-off events. Drums are all on "Channel
    # 10" (1-indexed!), so the commands are 144+9=153 for note on and 128+9=137
    # for note off. Also, the note-off velocities are all 4; this usually makes
    # no difference, & is a value I found in a MIDI file I saw somewhere once.
    events = np.array([[s*10000,153,p,v] for (p,v,s,d) in notes]
        + [[(s+d)*10000,137,p,4] for (p,v,s,d) in notes], dtype=int)
    events = events[events[:,0].argsort()]  #  sort by event time

    # Now we need to delete any note-off events that were intended to switch
    # off notes after which other notes have already switched on (on the same
    # "pitch," i.e., drum). That is, if I have two notes on the crash cymbal,
    # each nominally 2 seconds long, but actually only 1.5 seconds apart, then
    # the current event table will list note-on events at t=0 and 1.5, and
    # note-off events at t=2 and 3.5, but we want to NOT SEND the note-off at
    # t=2, since that would prematurely kill the SECOND crash note.
    to_remove = np.full(events.shape[0], False)
    for (k,p) in enumerate(events[:,2]):
        to_remove[k] = (events[k,1]==137) and \
            (np.sum(np.logical_and(events[:k+1,1]==153,events[:k+1,2]==p)) >
            np.sum(np.logical_and(events[:k+1,1]==137,events[:k+1,2]==p)))
    events = events[~to_remove,:]

    databytes += [byte for (deltaT, event) in
        zip(np.diff(np.concatenate(([0],events[:,0]))), events[:,1:])
        for byte in encode_var_length(deltaT)+event.tolist()]

    databytes += [0, 255, 47, 0]  #  end of track

    with open(filename, "wb") as fid:
        # Header chunk:
        fid.write(bytes([77,84,104,100,0,0,0,6,0,0,0,1] + encode_int(5000,2)))
        # Track chunk:
        fid.write(bytes([77,84,114,107] + encode_int(len(databytes),4) +
            databytes))

def encode_int(n,b):
    # MIDI sometimes requires numeric encodings of fixed length (2 or 4)
    return list((n).to_bytes(b,'big'))

def encode_var_length(n):
    # returns MIDI-specific variable-byte-length encoding of n
    return list(var_len_iterator(n))

def var_len_iterator(inp, continuing_flag=None):
  if inp>127:
    yield from var_len_iterator(inp>>7,1)
  yield (0 if (continuing_flag is None) else 128) + (inp & 127)

##  "USER INTERFACE"  #########################################################

def parse_script(file_name):
    lexer = AbomLexer()
    parser = AbomParser()
    with open(file_name, "r") as fid:
        script = fid.read()
    parser.parse(lexer.tokenize(script))

def parse_string(ab_code):
    lexer = AbomLexer()
    parser = AbomParser()
    parser.parse(lexer.tokenize(ab_code))

def main():
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage:\n  Abominable SCRIPT.TXT\nwhere SCRIPT.TXT contains " +
            "the tablature and other instructions.")
    else:
        parse_script(sys.argv[1])

if __name__ == '__main__':
    main()

##  TO-DO LIST  ###############################################################
"""
Comment the code more thoroughly
reduce imports?
PyInstaller? Result seems very bloated.
Deal with FOSS stuff:
  licenses (for this & sly &c.)
    sly license: https://github.com/dabeaz/sly/blob/master/LICENSE
Document this for users:
  how to user any possible PyInstaller version
  how to write and compile your script
  how to write your own drums in Python
  how to help develop the project (code quality, UI, sounds, functionality)
seek help:
  Use the appropriate exceptions
  Resolve the "shift/reduce conflicts"
MusicXML output?
MIDI output?
Swing?
Added to pip?
"""
