"""Separate WAV and missing-video decode checks using synthetic PCM audio."""
import hashlib,tempfile,unittest,wave
from pathlib import Path
import numpy as np
from jevaffect.media import decode_video

class SeparateAudioTests(unittest.TestCase):
    def test_missing_video_keeps_real_audio_and_marks_vision_absent(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            wav=folder/'sameclip.wav'
            with wave.open(str(wav),'wb') as output:
                output.setnchannels(1);output.setsampwidth(2);output.setframerate(16000)
                samples=(np.sin(np.arange(1600)*0.03)*1000).astype('<i2')
                output.writeframes(samples.tobytes())
            missing=folder/'sameclip.mp4'
            result=decode_video(missing,audio_path=wav,allow_missing_video=True,fps=2,size=32)
            self.assertEqual(result['metadata']['video_status'],'missing_file')
            self.assertEqual(result['metadata']['audio_status'],'present')
            self.assertEqual(result['metadata']['audio_samples'],1600)
            self.assertEqual(result['frames'].shape,(0,32,32,3))
            self.assertIsNone(result['metadata']['source_sha256'])
            self.assertEqual(result['metadata']['audio_source_sha256'],hashlib.sha256(wav.read_bytes()).hexdigest())
            other=folder/'other.wav'
            other.write_bytes(wav.read_bytes())
            with self.assertRaises(ValueError):
                decode_video(missing,audio_path=other,allow_missing_video=True)

if __name__=='__main__':unittest.main()
