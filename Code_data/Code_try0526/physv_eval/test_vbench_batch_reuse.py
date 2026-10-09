import unittest
from unittest.mock import patch
from pathlib import Path
from .vbench_official import OfficialVBenchRunner

class BatchReuseTests(unittest.TestCase):
    def test_one_call_order_and_scale(self):
        runner=OfficialVBenchRunner()
        payload=dict(raw_results=[dict(video_path='/tmp/b.mp4',video_results=72),dict(video_path='/tmp/a.mp4',video_results=81)],result_json='r',full_info_json='i',output_path='o')
        with patch.object(runner,'score_batch',return_value=payload) as call:
            rows=runner.score_cases_individually(['/tmp/a.mp4','/tmp/b.mp4'],dimension='imaging_quality')
            self.assertEqual(call.call_count,1)
            self.assertEqual([r['score'] for r in rows],[.81,.72])
            self.assertEqual([r['raw_results'][0]['video_results'] for r in rows],[81,72])
    def test_empty_does_not_load(self):
        runner=OfficialVBenchRunner()
        with patch.object(runner,'score_batch') as call:
            self.assertEqual(runner.score_cases_individually([],dimension='subject_consistency'),[])
            call.assert_not_called()
    def test_missing_result_fails(self):
        runner=OfficialVBenchRunner()
        with patch.object(runner,'score_batch',return_value=dict(raw_results=[])):
            with self.assertRaises(RuntimeError):runner.score_cases_individually(['/tmp/a.mp4'],dimension='subject_consistency')
    def test_duplicate_result_fails(self):
        runner=OfficialVBenchRunner();row=dict(video_path='/tmp/a.mp4',video_results=.8)
        with patch.object(runner,'score_batch',return_value=dict(raw_results=[row,row])):
            with self.assertRaises(RuntimeError):runner.score_cases_individually(['/tmp/a.mp4'],dimension='subject_consistency')

if __name__=='__main__':unittest.main()
