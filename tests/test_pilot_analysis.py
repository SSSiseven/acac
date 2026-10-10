"""AUC must share actual observed support, with no invented initial return."""
import unittest
from analysis.summarize_pilot import curve_auc,pooled_rate

class PilotAnalysisTests(unittest.TestCase):
    def test_interpolated_common_interval(self):
        auc,normalized=curve_auc([10,20,30],[0,10,20],15,25)
        self.assertAlmostEqual(auc,100)
        self.assertAlmostEqual(normalized,10)

    def test_reject_missing_support_and_duplicate_steps(self):
        with self.assertRaises(ValueError):curve_auc([10,20],[0,1],0,20)
        with self.assertRaises(ValueError):curve_auc([10,10],[0,1],10,20)

    def test_pooled_counts_weight_actual_eval_episode_counts(self):
        rows=[{"Eval/NumeratorAverage":"1","Eval/DenominatorAverage":"2",
               "Eval/EpisodeCountAverage":"20"},
              {"Eval/NumeratorAverage":"0","Eval/DenominatorAverage":"8",
               "Eval/EpisodeCountAverage":"24"}]
        self.assertAlmostEqual(pooled_rate(rows,"Numerator","Denominator"),20/232)

    def test_pooled_counts_not_mean_of_episode_ratios(self):
        rows=[{"Eval/NumeratorAverage":"1","Eval/DenominatorAverage":"2"},
              {"Eval/NumeratorAverage":"0","Eval/DenominatorAverage":"8"}]
        self.assertAlmostEqual(pooled_rate(rows,"Numerator","Denominator"),0.1)

if __name__=="__main__":unittest.main()
