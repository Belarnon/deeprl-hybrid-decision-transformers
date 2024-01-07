using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using GameLogic;

namespace PKB.App
{

    /// <summary>
    /// The <see cref="GameManager"/> holds the game state and manages the game loop.
    /// </summary>
    public class GameManager : MonoBehaviour
    {
        #region Inspector Interface

        /// <summary>
        /// The maximum number of blocks that can be selected for placement.
        /// </summary>
        [Tooltip("The maximum number of blocks that can be selected for placement.")]
        [SerializeField]
        private int numberOfBlocksToSelect = 3;

        /// <summary>
        /// The size of the grid in blocks.
        /// </summary>
        [Tooltip("The size of the grid in blocks.")]
        [SerializeField]
        private Vector2Int gridSize = new(10, 10);

        /// <summary>
        /// The block atlas that contains all available blocks in the game.
        /// </summary>
        [Tooltip("The block atlas that contains all available blocks in the game.")]
        [SerializeField]
        private BlockAtlasScriptableObject blockAtlas;
        /// <summary>
        /// This event is invoked every time the game state changes.
        /// </summary>
        public UnityEvent onGameStep;

        /// <summary>
        /// This event is invoked when the game is over (i.e. no more blocks can be placed).
        /// </summary>
        public UnityEvent onGameOver;

        /// <summary>
        /// This event is invoked when the game is reset.
        /// </summary>
        public UnityEvent onGameReset;

        /// <summary>
        /// This event is invoked when the grid size changes.
        /// </summary>
        public UnityEvent<Vector2Int> onGridSizeChanged;

        /// <summary>
        /// This event is invoked when the number of blocks to select changes.
        /// </summary>
        public UnityEvent<int> onNumberOfBlocksToSelectChanged;

        /// <summary>
        /// This event is invoked when the block atlas changes.
        /// </summary>
        public UnityEvent<BlockAtlasScriptableObject> onBlockAtlasChanged;

        #endregion

        #region  Internal State

        private GameInstance gameInstance;

        #endregion

        #region Unity Lifecycle

        private void Start()
        {
            gameInstance = new GameInstance(gridSize.x, gridSize.y, numberOfBlocksToSelect, blockAtlas);
            onGridSizeChanged.Invoke(gridSize);
            onNumberOfBlocksToSelectChanged.Invoke(numberOfBlocksToSelect);
            onBlockAtlasChanged.Invoke(blockAtlas);
            onGameReset.Invoke();
        }

        #endregion

        #region Public Interface

        /// <summary>
        /// Sets the block with the given index at the center block to the given coordinates.
        /// </summary>
        /// <param name="blockIndex">The index of the block to place.</param>
        /// <param name="coordinates">The coordinates of where to place the center block of the tile.</param>
        public void PutBlock(int blockIndex, Vector2Int coordinates)
        {
            Debug.Log($"Putting block {blockIndex} at {coordinates}.");
            bool success = gameInstance.putBlock(blockIndex, coordinates);
            if (!success)
            {
                Debug.LogError($"Failed to put block {blockIndex} at {coordinates}.");
                return;
            }
            UpdateState();
        }

        /// <summary>
        /// Attempts to put the block with the given index at the center block to the given coordinates.
        /// </summary>
        /// <param name="blockIndex">The index of the block to place.</param>
        /// <param name="coordinates">The coordinates of where to place the center block of the tile.</param>
        /// <returns>True if the block was successfully placed, false otherwise.</returns>
        public bool AttemptPutBlock(int blockIndex, Vector2Int coordinates)
        {
            Debug.Log($"Attempting to put block {blockIndex} at {coordinates}.");
            bool success = gameInstance.putBlock(blockIndex, coordinates);
            if (success)
            {
                UpdateState();
            }
            return success;
        }

        /// <summary>
        /// Resets the game to its initial state.
        /// </summary>
        public void ResetGame()
        {
            gameInstance.reset();
            onGameReset.Invoke();
        }

        /// <summary>
        /// Resets the game to its initial state with the given parameters.
        /// </summary>
        /// <param name="numberOfBlocksToSelect">The new number of blocks to select.</param>
        /// <param name="gridSize">The new grid size.</param>
        /// <param name="blockAtlas">The new block atlas.</param>
        public void ResetGame(int numberOfBlocksToSelect, Vector2Int gridSize, BlockAtlasScriptableObject blockAtlas)
        {
            this.numberOfBlocksToSelect = numberOfBlocksToSelect;
            this.gridSize = gridSize;
            this.blockAtlas = blockAtlas;
            gameInstance.reset(gridSize.x, gridSize.y, numberOfBlocksToSelect, blockAtlas);
            onGameReset.Invoke();
            onGridSizeChanged.Invoke(gridSize);
            onNumberOfBlocksToSelectChanged.Invoke(numberOfBlocksToSelect);
            onBlockAtlasChanged.Invoke(blockAtlas);
        }

        /// <summary>
        /// Returns the current score.
        /// </summary>
        /// <returns>The current score.</returns>
        public int GetCurrentScore()
        {
            return gameInstance.getScore();
        }
        
        /// <summary>
        /// Returns the current placement of blocks (true = block, false = empty)
        /// </summary>
        /// <returns>The current placement of blocks.</returns>
        public bool[,] GetCurrentPlacement()
        {
            return gameInstance.getGrid();
        }

        /// <summary>
        /// Returns the list of the currently available blocks for selection.
        /// </summary>
        /// <returns>The list of the currently available blocks for selection.</returns>
        public List<BlockScriptableObject> GetAvailableBlocks()
        {
            return gameInstance.getGivenBlocks();
        }

        /// <summary>
        /// Returns the number of blocks that can be selected in total (regardless of how many are currently available).
        /// </summary>
        /// <returns>The number of blocks that can be selected in total.</returns>
        public int GetNumberOfBlocksToSelect()
        {
            return numberOfBlocksToSelect;
        }

        /// <summary>
        /// Returns the size of the grid.
        /// </summary>
        /// <returns>The size of the grid.</returns>
        public Vector2Int GetGridSize()
        {
            return gridSize;
        }

        #endregion

        #region Helpers

        private void UpdateState() {
            if (gameInstance.gameIsSolvable()) {
                onGameStep.Invoke();
            } else {
                onGameOver.Invoke();
            }
        }

        #endregion
    }
}
