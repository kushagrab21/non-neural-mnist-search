"""Planted synthetic controls; no trained parameters or MNIST images."""
SOURCES={
    'stateless': 'def predict(image): return int(image[0,0])%10\n',
    'call_count': 'calls=0\ndef predict(image):\n    global calls\n    calls+=1\n    return calls%10\n',
    'previous_image': 'previous=0\ndef predict(image):\n    global previous\n    result=previous\n    previous=int(image[0,0])%10\n    return result\n',
    'delayed_512': 'calls=0\ndef predict(image):\n    global calls\n    calls+=1\n    return (int(image[0,0])+(calls>512))%10\n',
}
