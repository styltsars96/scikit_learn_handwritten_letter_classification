from sys import argv

from Classifier_Test import test_classifier

if __name__ == '__main__':
    if len(argv) == 1:
        print("Need arguements...")

    classifier = ""
    normalize = False

    for arg in argv:
        cur_arg = arg.lower()
        if cur_arg == "main.py":
            continue
        # collect arguments...
        elif "classifier=" in cur_arg:
            classifier = cur_arg.split("=")[1]
        elif "--normalized_matrix" in cur_arg:
            normalize = True
        else:
            print("Invalid arguement:", arg)

    if classifier == "":
        print("Select a classifier!\nGive argument: classifier=<your_choice>"
              + "\nUsing Default (KNN)...")
        test_classifier("letter-recognition.data", normalization=normalize)
    else:
        test_classifier("letter-recognition.data", classif=classifier,
                        normalization=normalize)
