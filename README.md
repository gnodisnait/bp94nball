# The Function

A demo to show that normal back-propagation (BP) method is not(9) for(4) training nball embeddings. That is, BP can not achieve zero energy cost for embedding tree/forest structures into word embeddings.

# Description

given a very small semantic relations as follow.
```
rock_1 is_a material, is_not rock_2
stone is_a material, is_not jazz
basalt is_a material, is_not communication
material is_a substance, is_not event
substance is_a entity, is_not music

rock_2 is_a music, is_not material
pop is_a music, is_not  substance
jazz is_a music, is_not basalt
music is_a communication, is_not entity
communication is_a event, is_not entity
```

these entities are initialised as two dimensional balls as follows.
```
rock_1 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 10, 3]
stone = [np.cos(np.pi / 4 + np.pi / 100), np.sin(np.pi / 4 + np.pi / 100), 10, 3]
basalt = [np.cos(np.pi / 4 - np.pi / 100), np.sin(np.pi / 4 - np.pi / 100), 10, 3]
material = [np.cos(np.pi / 4 + np.pi / 30), np.sin(np.pi / 4 + np.pi / 30), 10, 3]
substance = [np.cos(np.pi / 4 + np.pi / 15), np.sin(np.pi / 4 + np.pi / 15), 10, 3]
entity = [np.cos(np.pi / 4 - np.pi / 200), np.sin(np.pi / 4 - np.pi / 200), 10, 3]

rock_2 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 10, 3]
pop = [np.cos(np.pi / 4 + np.pi / 50), np.sin(np.pi / 4 + np.pi / 50), 10, 3]
jazz = [np.cos(np.pi / 4 - np.pi / 50), np.sin(np.pi / 4 - np.pi / 50), 10, 3]
music = [np.cos(np.pi / 3), np.sin(np.pi / 3), 10, 3]
communication = [np.cos(np.pi / 3 + np.pi / 50), np.sin(np.pi / 3 + np.pi / 50), 10, 3]
event = [np.cos(np.pi / 3 - np.pi / 50), np.sin(np.pi / 3 - np.pi / 50), 10, 3]
```
Each ball is represented by a list with four elements, ```[np.cos(x), np.sin(x), l, r]```,
where ```x, l, r``` representing angle, length of the center vector, and the radius


# Quick start
```
$ git clone https://github.com/gnodisnait/bp94nball.git
$ cd bp94nball
bp94nball $ conda create --name BP94NBALL
bp94nball $ source activate BP94NBALL
(BP94NBALL) bp94nball $ make init
```

# Run demo
```
(BP94NBALL) bp94nball $ make show
```
A result is illustrated as below.

![](https://github.com/gnodisnait/bp94nball/blob/master/pic/bp94nball.png)

# Run python command line with  restart
```
(BP94NBALL) bp94nball $ python3 main.py --restart 1
```

# Run python command line without  restart
```
(BP94NBALL) bp94nball $ python3 main.py --restart 0
```
