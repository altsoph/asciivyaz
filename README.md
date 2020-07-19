# ASCIIvyaz
![](https://raw.githubusercontent.com/altsoph/asciivyaz/master/gallery/logo_square.png)
# tldr
I tried to automate the formation of ligatures in style of Slavic calligraphy, called [Vyaz](https://en.wikipedia.org/wiki/Vyaz).
A gallery with examples of results can be [found here](https://github.com/altsoph/asciivyaz/tree/master/gallery). 
Details below.
## intro
There is an old Slavic calligraphy tradition, called 'Russian Vyaz' or just 'Vyaz'.
It's unique in terms of special forms and rules of ligature generation.
For example, take a look at works of a [contemporary artist Viktor Pushkarev](https://www.behance.net/mynameisviktor).

Once, a colleague of mine, [Anna Shishlyakova](https://twitter.com/ashlkv) drew a nice piece of Vyaz and since then I kept thinking if I can automatically generate such ligatures. Several years have passed and I finally found a weekend to make something like this:

![the original drawing by Anna on the left side and the generated vyaz on the right side](https://raw.githubusercontent.com/altsoph/asciivyaz/master/gallery/greplogs.png)
*the original drawing by Anna on the left side and the generated vyaz on the right side*

##general description
The current implementation consists of several components:
* a configuration file that determines the size of symbols, kerning and ligature assembly parameters (as an example, I give a pack of several different configuration files with crafted parameters)
* a font geometry file now contains Latin and Cyrillic symbols, plus Arabic numerals, it can be expanded
* a script that implements several rendering algorithms:
  * the **plain** method just prints letters, almost like [figlet](https://en.wikipedia.org/wiki/FIGlet),
  * the **greedy** algorithm, that, if possible, attaches letters to the current ligature one by one, from left to right,
  * the **random** method, that assembles ligatures in a random sequence,
  * the **random_best** meta-algorithm that runs the *random* method several times and selects the optimal result (in terms of the width and distortion minimization).

The current version renders the result in ASCII/unicode graphics, however, the whole core is implemented in a vector logic, so it is possible to rewrite a renderer to produce something like svg or high-resolution bitmap.
## details
### requirements
All you need is python 3.x and yaml library.
### command line options
There are a few of them:

```
  -h, --help           show this help message and exit
  --config CONFIG      config file
  --geometry GEOMETRY  font geometry file
  --text TEXT          text to draw
  --seed SEED          random seed
  --method METHOD      method to use
```
### core logic
In case you want to add new symbols and/or change how the font looks, you should understand several principles:
* Each symbol consists of a set of anchors and some edges between them.
* The position of each anchor is given in terms of a simple grid:
  * for a horizontal offset I used the zero-based index  of "vertical", for example, 'I' has only one vertical, 'U' has two verticals, and 'W' has three vertical lines (note, 'Z' still has two verticals),
  * for a vertical position I used this scale of 5 positions: b[ottom] < d[own] < m[iddle] < u[p] < t[op].
* An edge is defined by three parameters:
  * the first anchor,
  * the second anchor,
  * the constraint on a slope of the edge:
    * '=' means both anchors should be strictly on the same vertical position,
    * '<' means the first anchor should be lower than the second, etc.
* While merging two symbols into ligature the script checks if a left-most vertical of the right symbol conflicts wit a right-most vertical of the left symbol. If so, it tries to distort them somehow to avoid any conflicts. There are two types of distortion implemented:
  * if it's possible to resize the only one vertical of symbol to remove conflicts without violation of edges constraints, it's done.
  * Overwise, the whole symbol could be resized to avoid the collision, but it's bad to resize too much,
  * If nothing helps, the script just puts these two symbols side-by-side separately.
### configuration file options
```yaml
# this block specifies a char size
h: 20   # height
w: 5    # distance between verticals of letter
dx: 2   # horizontal boldness of lines
dy: 1   # vertical boldness of lines
# anchors positions
t: 20   # top anchor postision
u: 15   # upper anchor postision
m: 10   # middle anchor postision
d: 6    # down anchor postision
b: 0    # bottom anchor postision
# symbols
a: ▒    # use this character to draw anchors
e: █    # use this character to draw edges
# magic parameters (modify them with care and courage)
vc:  1  # margin size for intersection checks
pad: 3  # interletter padding
f: 6    # maximum vertical displacement, should be le than w/2
sq: 0   # squeeze (0/1)
dp: 1   # antidistortion weight
rtr: 16 # num of retries for random layout
fsq: 1  # final compression (-1 = off, other value -- padding we want to keep)
```
## few more words
Despite the output of the asciivyaz script usually looks okay, consider to use it in conjunction with some other tools:
* try to use [lolcat](https://github.com/busyloop/lolcat) to colorize the output just in a terminal,
* try to use some online renderers, like [carbon](https://carbon.now.sh/) to make snippets look better.

![](https://github.com/altsoph/asciivyaz/blob/master/gallery/magic_text.png?raw=true)
