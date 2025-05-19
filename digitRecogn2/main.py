from digitRecogn2.pipeline import run_inference

totalDets = 0
corrDets = 0

for i in range(0, 10):
    print(i)
    path = f'C:/UTCN/an3/sem2/pi/proj/mnistDigitRecogniser/digitRecogn2/images/{i}s.png'
    dets = run_inference(path)
    totalDets += len(dets)
    for det in dets:
        if det == i:
            corrDets += 1

    print('\n')

print(corrDets/totalDets)