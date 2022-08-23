### Panoramic Image Stitching and Registration

<!-- PROJECT SHIELDS -->
<!--
-->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#Registration">Registration</a></li>
        <li><a href="#Stitching">Stitching</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#Creating Panoramic Images">Creating Panoramic Images</a></li>
      </ul>
    </li>
  </ol>
</details>

## About The Project

In this project, I developed the tools required to perform a panoramic image 
stitching, using a method called "Stereo Mosaicing".
In order to do so, I implemented some essential algorithms
We can split the process into 2 different steps:

### Registration: 
The geometric transformation between each consecutive image pair is found by detecting
Harris feature points, extracting their MOPS-like descriptors, matching these descriptors
between the pair and fitting a rigid transformation that agrees with a large set of inlier matches
using the RANSAC algorithm.
### Stitching:
Combining strips from aligned images into a sequence of panoramas. Global motion will be compensated,
and the residual parallax, as well as other motions will become visible.

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

1. clone the project into your computer
2. make sure packages are installed using 'requirements.txt'

## Create Panoramic image using the test videos:
In order to test the algorithm's performance, and observe its result,
I provided you some examples you can try by yourself:
- 'home.mp4' 
- 'iguazu.mp4'
- 'boat.mp4'

Here's an example for the output of the video provided in this project:

Result for 'boat.mp4':
![Alt text](boat.png?raw=true "Title")
Result for 'iguazu.mp4':
![Alt text](iguazu.png?raw=true "Title")
Result for 'home.mp4':
![Alt text](test_result.png?raw=true "Title")



## You can do it!
I guess that you want to try it using your own video, right? to do so, follow the instructions below:

1. How to record the video:
Reminder: this algorithm is working under the assumption that every two overlapping frames can be related to each other by applying a rigid transformation, i.e, transformation which includes only rotation and translation.
Hence, in order to make this work, you need to take a video, using only a rotation and\or translation movements while recording.
another thing to pay attention to is the length of the video - take your time, and let the video be at least as long as the provided ones
Lastly, make sure that you don't move too fast, so the algorithm can match points of overlapping frames correctly.

2. Feeding it into the program:
- Copy the video into a directory called 'videos' which is located in the project's main directory.
- In the my_panorama file, you'll find 


Hope you find it useful!

<!-- MARKDOWN LINKS & IMAGES -->

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/tzlil-ovadia
