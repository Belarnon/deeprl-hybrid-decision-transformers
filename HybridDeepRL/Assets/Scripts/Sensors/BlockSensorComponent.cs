using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;
using PKB.App;
using GameLogic;

namespace PKB.Sensors
{
    public class BlockSensorComponent : GridRenderSensorComponent
    {

        #region Inspector Interface

        /// <summary>
        /// A reference to the game manager to grab the game state from.
        /// </summary>
        [Tooltip("A reference to the game manager to grab the game state from.")]
        [SerializeField]
        private GameManager gameManager;

        /// <summary>
        /// The block index to render.
        /// </summary>
        [Tooltip("The block index to render.")]
        [SerializeField]
        private int blockIndex;

        [Tooltip("Maximum dimension of a single block.")]
        [SerializeField]
        private Vector2Int maxBlockDimension;

        #endregion

        #region Internal State

        private bool[,] emptyBlockGrid;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            Debug.Assert(gameManager != null, "No game manager reference set.");

            InitTextures(maxBlockDimension);
            emptyBlockGrid = new bool[maxBlockDimension.x, maxBlockDimension.y];

            gameManager.onGameStep.AddListener(OnBoardStateChanged);
            gameManager.onGameReset.AddListener(OnBoardStateChanged);
        }

        #endregion

        #region GameManager Callbacks

        private void OnBoardStateChanged()
        {
            BlockScriptableObject block = gameManager.GetAvailableBlocks()[blockIndex];
            if (block == null)
            {
                FillTexture(emptyBlockGrid);
            }
            else
            {
                FillTexture(block.getGrid());
            }
            UpdateRenderTexture();
        }

        #endregion
    }
}
