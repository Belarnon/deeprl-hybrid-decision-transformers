using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;
using PKB.App;

namespace PKB.Sensors
{

    /// <summary>
    /// The <see cref="BoardSensorComponent"/> is a <see cref="GridRenderSensorComponent"/> that
    /// renders the game board for use as an observation by an agent.
    /// </summary>
    [AddComponentMenu("PKB/Sensors/Board Sensor")]
    public class BoardSensorComponent : GridRenderSensorComponent
    {
        #region Inspector Interface

        /// <summary>
        /// A reference to the game manager to grab the game state from.
        /// </summary>
        [Tooltip("A reference to the game manager to grab the game state from.")]
        [SerializeField]
        private GameManager gameManager;

        #endregion

        #region Internal State
        private Vector2Int lastGridSize;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            Debug.Assert(gameManager != null, "No game manager reference set.");

            lastGridSize = gameManager.GetGridSize();
            InitTextures(lastGridSize);

            gameManager.onGridSizeChanged.AddListener(OnBoardSizeChanged);
            gameManager.onGameStep.AddListener(OnBoardStateChanged);
            gameManager.onGameReset.AddListener(OnBoardStateChanged);
        }

        #endregion

        #region GameManager Callbacks

        private void OnBoardSizeChanged(Vector2Int newSize)
        {
            if (newSize != lastGridSize)
            {
                InitTextures(newSize);
                lastGridSize = newSize;
            }
        }

        private void OnBoardStateChanged()
        {
            bool[,] board = gameManager.GetCurrentPlacement();
            FillTexture(board);
            UpdateRenderTexture();
        }

        #endregion
    }
}
