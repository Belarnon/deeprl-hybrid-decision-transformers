using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PKB.App;

namespace PKB.UI
{

    /// <summary>
    /// A <see cref="GridUpdater"/> is a component that updates the grid UI when the game state changes.
    /// </summary>
    [RequireComponent(typeof(GameGrid))]
    public class GridUpdater : MonoBehaviour
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

        private GameGrid gameGrid;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            gameGrid = GetComponent<GameGrid>();
            gameManager.onGameStep.AddListener(OnGameStep);
            gameManager.onGameOver.AddListener(OnGameStep);
        }

        #endregion

        #region Public Interface

        public void OnGameStep()
        {
            gameGrid.UpdateGridCells(gameManager.GetCurrentPlacement());
        }

        #endregion
    }
}
