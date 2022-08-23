import os
import time

from panorama_stitching import PanoramicVideoGenerator


def main():
    inpt = input("Hello and welcome! \nIf you wish to run the test example, press enter to continue.\n"
          "else, insert the filename you wish to run the test on:\n")
    if inpt == '\n':
        experiments = ['home.mp4', 'iguazu.mp4', 'boat.mp4']
    else:
        experiments = [inpt]

    for experiment in experiments:
        try:
            exp_no_ext = experiment.split('.')[0]
            os.system('mkdir dump')
            os.system(('mkdir ' + str(os.path.join('dump', '%s'))) % exp_no_ext)
            os.system(
                ('ffmpeg -i ' + str(os.path.join('videos', '%s ')) + str(os.path.join('dump', '%s', '%s%%03d.jpg'))) % (
                    experiment, exp_no_ext, exp_no_ext))

            s = time.time()
            panorama_generator = PanoramicVideoGenerator(os.path.join('dump', '%s') % exp_no_ext, exp_no_ext, 2100)
            panorama_generator.align_images(translation_only=experiment in experiment)
            panorama_generator.generate_panoramic_images(9)
            print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))

            panorama_generator.save_panoramas_to_video()
        except FileNotFoundError:
                print('File not found, please try again.')


if __name__ == '__main__':
  main()
