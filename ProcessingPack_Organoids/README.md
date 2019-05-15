# ProcessingPack [Draft, Untested]
!["One Piece at a Time"](/resources/chip-working.png)

## Overview
ProcessingPack is a set of classes and functions to execute MITOMI image processing (feature finding and quantification). 

It is designed around the following paradigm:
 - Imaging of a *ChipImage* is *single* or *series*
 	- A given *ChipImage* is a 2-D array of *Stamps*
 	- A *Stamp* contains feature with attributes. Principle features are:
 		- *Chamber*: the primary reaction chamber
 		- *Button*: the contact area of the button on the slide

### Architecture
- TODO

#### Templates and Examples
- TODO

#### Requirements
opencv-python==3.4.0.12<br>
numpy==1.14.1<br>
scikit-image==0.13.1<br>
tqdm==4.23.4<br>
pandas==0.22.0<br>

## Tasks
- [ ] Clean up comments
- [ ] Generate template workup(s)
- [ ] Additional Testing
