import Abominable as ab

# Define a custom new drum sound, by specifying a few parameters. The format
# required is that you define a Python function and insert it into the
# dictionary ab.drum_dict under the key drum_name_in_lowercase. That function
# can input whatever parameters you like (and they are then used in Abominable
# scripts when defining tab lines), and it should return a dictionary,
# specifying values for "note parameters." Some of the "note parameters" are
# scalars (floats), and others are functions. Best to see an example:

def wotsit_func(fund_freq=50, exponent=0.85):
    return dict(
        freqs = lambda k: fund_freq*(k+1)**exponent,
        sharpness = 40,
        a0 = 0.01,
        ap = 0.01,
        decay = lambda k: 30*(1+3*(k%3)),
        peak = lambda k: ab.rng.random()**5,
        close = lambda k: 0.02+k*0.02)

# Now include the "drum" you've just created in Abominable's dictionary, along
# with the built-in drums:
ab.drum_dict["wotsit"] = wotsit_func

# Because you've provided default values for the two parameters, you can either
# specify those values in the Abominable script or not. (If you're only using
# one version of the sound, then there's no need to allow parameters.

# Now that we've done that, we could write a .txt file as usual, and then run
# it from here with the command
#   parse_script(demo3.txt).
# Since we need this file anyway, though, it's probably easier to keep
# everything in here, and you can do that via the equivalent command
# parse_string, as follows. First, we create one big string with the tablature
# and other Abominable code:

ab_code = """
tempo 210

tablines =
hihat 0.04s
wotsit 1
bass 1

tab bar =
11111111|
--1--1--|
1---1---|

sequence beat =
bar x4

export beat custom_instrument.wav
"""

ab.parse_string(ab_code)
