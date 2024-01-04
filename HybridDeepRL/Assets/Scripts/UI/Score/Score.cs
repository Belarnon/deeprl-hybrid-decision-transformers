using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

namespace PKB.UI
{
    /// <summary>
    /// A <see cref="Score"/> panel can be used to display the current score.
    /// </summary>
    [AddComponentMenu("PKB/UI/Score")]
    public class Score : MonoBehaviour
    {
        
        #region Inspector Interface

        /// <summary>
        /// The text mesh pro component that displays the score.
        /// </summary>
        [Tooltip("The text mesh pro component that displays the score.")]
        [SerializeField]
        private TextMeshProUGUI scoreText;

        #endregion

        #region Internal State

        /// <summary>
        /// The current score.
        /// </summary>
        private int score = 0;

        #endregion

        #region Public Interface

        /// <summary>
        /// Sets the score to the given value.
        /// </summary>
        /// <param name="value">The new score.</param>
        public void SetScore(int value)
        {
            score = value;
            scoreText.text = score.ToString();
        }
        
        #endregion
    }
}
