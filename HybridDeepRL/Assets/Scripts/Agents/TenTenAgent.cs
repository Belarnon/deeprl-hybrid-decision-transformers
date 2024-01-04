using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using PKB.App;
using GameLogic;

namespace PBK.Agents
{

    /// <summary>
    /// The <see cref="TenTenAgent"/> implements the ML-Agents API for the 1010! game.
    /// </summary>
    public class TenTenAgent : Agent
    {

        #region Inspector Interface

        /// <summary>
        /// Reference to the game simulation.
        /// </summary>
        [Tooltip("Reference to the game simulation.")]
        [SerializeField]
        private GameManager gameManager;


        /// <summary>
        /// The size of the observation grid for a single block.
        /// </summary>
        [Tooltip("The size of the observation grid for a single block.")]
        [SerializeField]
        private Vector2Int blockObservationSize = new(5, 5);

        #endregion

        #region Internal State

        private int lastScore = 0;

        private int maxPossibleScoreDelta = 0;

        private int numberOfAvailableBlocks = 0;

        #endregion

        #region Unity Lifecycle

        private void Start()
        {
            CheckDependencies();
            ComputeMaxPossibleScoreDelta();
            gameManager.onGameStep.AddListener(OnGameStep);
            gameManager.onGameOver.AddListener(OnGameOver);
        }

        #endregion

        #region Unity ML-Agents API

        public override void OnEpisodeBegin()
        {
            ResetGame();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            bool[,] currentGrid = gameManager.GetCurrentPlacement();
            FillSensorWithGrid(sensor, currentGrid);
            int selectableBlocks = gameManager.GetNumberOfBlocksToSelect();
            List<BlockScriptableObject> availableBlocks = gameManager.GetAvailableBlocks();
            numberOfAvailableBlocks = availableBlocks.Count;
            for (int i = 0; i < selectableBlocks; i++)
            {
                if (i < availableBlocks.Count)
                {
                    BlockScriptableObject block = availableBlocks[i];
                    FillSensorWithBlock(sensor, block);
                }
                else
                {
                    FillSensorWithEmptyBlock(sensor);
                }
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            int blockIndex = actions.DiscreteActions[0];
            if (blockIndex >= numberOfAvailableBlocks)
            {
                AddReward(-0.05f);
                return;
            }
            int x = actions.DiscreteActions[1];
            int y = actions.DiscreteActions[2];
            bool success = gameManager.AttemptPutBlock(blockIndex, new Vector2Int(x, y));
            if (!success)
            {
                AddReward(-0.05f);
            }
        }

        #endregion

        #region Game State Callbacks

        public void OnGameStep()
        {
            float reward = ComputeScoreReward(gameManager.GetCurrentScore());
            AddReward(reward);
        }

        public void OnGameOver()
        {
            AddReward(-1f);
            EndEpisode();
        }

        #endregion

        #region Helpers

        private void CheckDependencies()
        {
            Assert.IsNotNull(gameManager, "Game instance must be set.");
        }

        private void ResetGame()
        {
            gameManager.ResetGame();
            lastScore = 0;
        }

        private static void FillSensorWithGrid(VectorSensor sensor, bool[,] grid)
        {
            int width = grid.GetLength(0);
            int height = grid.GetLength(1);
            for (int x = 0; x < width; x++)
            {
                for (int y = height - 1; y >= 0; y--)
                {
                    sensor.AddObservation(grid[x, y]);
                }
            }
        }

        private static void FillSensorWithBlock(VectorSensor sensor, BlockScriptableObject block)
        {
            bool[,] grid = block.getGrid();
            FillSensorWithGrid(sensor, grid);
        }

        private void FillSensorWithEmptyBlock(VectorSensor sensor)
        {
            FillSensorWithGrid(sensor, new bool[blockObservationSize.x, blockObservationSize.y]);
        }

        private void ComputeMaxPossibleScoreDelta()
        {
            Vector2Int gridSize = gameManager.GetGridSize();
            maxPossibleScoreDelta = blockObservationSize.x * gridSize.x + blockObservationSize.y * gridSize.y;
        }

        private float ComputeScoreReward(int score)
        {
            int delta = score - lastScore;
            float normalizedScoreDelta = (float)delta / maxPossibleScoreDelta;
            lastScore = score;
            return normalizedScoreDelta;
        }

        #endregion
    }
}
