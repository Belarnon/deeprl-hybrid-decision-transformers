using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PKB.App;

namespace PKB.UI
{

    /// <summary>
    /// A <see cref="ScoreUpdater"/> is a component that updates the score when the game state changes.
    /// </summary>
    [AddComponentMenu("PKB/UI/Score Updater")]
    [RequireComponent(typeof(Score))]
    public class ScoreUpdater : MonoBehaviour
    {
        #region Inspector Interface

        /// <summary>
        /// The game manager that holds the game state.
        /// </summary>
        [Tooltip("The game manager that holds the game state.")]
        [SerializeField]
        private GameManager gameManager;

        #endregion

        #region Internal State

        /// <summary>
        /// The score component that is updated.
        /// </summary>
        private Score score;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            score = GetComponent<Score>();
            gameManager.onGameStep.AddListener(UpdateScore);
            gameManager.onGameReset.AddListener(UpdateScore);
        }

        #endregion

        #region Event Handlers

        /// <summary>
        /// Updates the score.
        /// </summary>
        private void UpdateScore()
        {
            score.SetScore(gameManager.GetCurrentScore());
        }

        #endregion
    }
}
