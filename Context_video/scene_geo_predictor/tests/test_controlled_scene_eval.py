"""Ensure single-variable intervention preserves the dynamic actor and camera."""
from dataclasses import asdict
import unittest
import numpy as np

from controlled_scene_eval import generator, make_case, VALUES, records


class ControlledTests(unittest.TestCase):
    def test_response_metrics(self):
        from evaluate_controlled_scene import response_metrics
        a=np.zeros((41,3)); b=a+np.array([.2,0,0])
        exact=response_metrics(a,b,a,b)
        self.assertAlmostEqual(exact['response_error_m'],0)
        self.assertAlmostEqual(exact['response_gain'],1)
        self.assertAlmostEqual(exact['response_cosine'],1)
        unchanged=response_metrics(a,a,a,b)
        self.assertAlmostEqual(unchanged['response_error_m'],.2)
        self.assertEqual(unchanged['response_gain'],0)
        self.assertIsNone(unchanged['response_cosine'])
        identical=response_metrics(a,a,a,a)
        self.assertIsNone(identical['response_gain'])
        self.assertIsNone(identical['gt_first_difference_frame'])

    def test_unique_eval_only_records(self):
        rows = records()
        self.assertEqual(len(rows),12)
        self.assertEqual(len({r['key'] for r in rows}),12)
        self.assertTrue(all(r['split']=='control_eval' for r in rows))

    def test_single_variable_geometry(self):
        e = generator()
        for family,values in VALUES.items():
            cases = [make_case(e,family,'fixed_key',v).blueprint for v in values]
            reference = cases[0]
            for case in cases[1:]:
                self.assertEqual(asdict(case.camera),asdict(reference.camera))
                self.assertEqual([asdict(o) for o in case.objects if o.dynamic],
                                 [asdict(o) for o in reference.objects if o.dynamic])
                for a,b in zip(reference.objects,case.objects):
                    x,y = asdict(a),asdict(b)
                    for field in ('position','size'):
                        x.pop(field); y.pop(field)
                    self.assertEqual(x,y)
                if family=='barrier':
                    a,b = reference.objects[-1],case.objects[-1]
                    self.assertEqual(a.position[1:],b.position[1:])
                    self.assertEqual(a.size,b.size)


if __name__=='__main__':
    unittest.main()
