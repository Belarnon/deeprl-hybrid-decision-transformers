using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using PKB.App;
using PKB.Recording;
using GameLogic;
using Unity.MLAgents.Policies;

namespace PBK.Agents
{

    /// <summary>
    /// The <see cref="TenTenAgent"/> implements the ML-Agents API for the 1010! game.
    /// </summary>
    public class TenTenAgent : RecordableAgent
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

        /// <summary>
        /// The multiplier for positive rewards (i.e. rewards for placing blocks).
        /// </summary>
        [Tooltip("The multiplier for positive rewards (i.e. rewards for placing blocks).")]
        [SerializeField]
        [Range(0f, 5f)]
        private float positiveRewardMultiplier = 1f;

        /// <summary>
        /// The multiplier for negative rewards (i.e. rewards for invalid actions).
        /// </summary>
        [Tooltip("The multiplier for negative rewards (i.e. rewards for invalid actions).")]
        [SerializeField]
        [Range(0f, 5f)]
        private float negativeRewardMultiplier = 1f;

        #endregion

        #region Internal State

        private int lastScore = 0;

        private int maxPossibleScoreDelta = 0;

        private int numberOfAvailableBlocks = 0;

        private bool gameIsOver = false;
        
        private UserInput userInput;

        private BehaviorParameters behaviorParameters;

        private bool isHeuristicAgent = false;

        #endregion

        #region Unity Lifecycle

        private void Start()
        {
            CheckDependencies();
            behaviorParameters = GetComponent<BehaviorParameters>();
            isHeuristicAgent = IsHeuristicAgent();
            ComputeMaxPossibleScoreDelta();
            numberOfAvailableBlocks = gameManager.GetNumberOfBlocksToSelect();
            gameManager.onGameStep.AddListener(OnGameStep);
            gameManager.onGameOver.AddListener(OnGameOver);
        }

        #endregion

        #region Public Interface
        
        /// <summary>
        /// Externally sets the user input to use in the next heuristic decision.
        /// </summary>
        /// <param name="blockIndex">The index of the block to place.</param>
        /// <param name="position">The position to place the block at.</param>
        public void SetUserInput(int blockIndex, Vector2Int position)
        {
            if (!isHeuristicAgent)
            {
                return;
            }

            userInput.blockIndex = blockIndex;
            userInput.position = position;
            userInput.hasUserInput = true;
            
            // We need to manually request a decision here, since the user input is not
            // processed until processed by the heuristic decision.
            RequestDecision();
        }

        /// <summary>
        /// Sets the currently recorded trajectory as ended and terminates the episode.
        /// </summary>
        public void MarkAndEndEpisode()
        {
            MarkTrajectoryEnd();
            EndEpisode();
        }

        #endregion

        #region Unity ML-Agents API

        public override void OnEpisodeBegin()
        {
            ResetGame();
            RequestDecisionIfNotHeuristic();
            MarkTrajectoryStart();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            bool[,] currentGrid = gameManager.GetCurrentPlacement();
            FillSensorWithGrid(sensor, currentGrid);
            BlockScriptableObject[] availableBlocks = gameManager.GetAvailableBlocks();
            for (int i = 0; i < numberOfAvailableBlocks; i++)
            {
                BlockScriptableObject block = availableBlocks[i];
                if (block != null)
                {
                    
                    FillSensorWithBlock(sensor, block);
                }
                else
                {
                    FillSensorWithEmptyBlock(sensor);
                }
            }
            CommitObservation(sensor);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
            if (userInput.hasUserInput)
            {
                discreteActions[0] = userInput.blockIndex;
                discreteActions[1] = userInput.position.x;
                discreteActions[2] = userInput.position.y;
                userInput.hasUserInput = false;
            }
            else
            {
                discreteActions[0] = -1;
                discreteActions[1] = -1;
                discreteActions[2] = -1;
            } 
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            int blockIndex = actions.DiscreteActions[0];
            int x = actions.DiscreteActions[1];
            int y = actions.DiscreteActions[2];
            if (IsNopAction(blockIndex, x, y))
            {
                SetReward(0f);
                RequestDecisionIfNotHeuristic();
                return;
            }
            bool success = gameManager.AttemptPutBlock(blockIndex, new Vector2Int(x, y));
            if (gameIsOver)
            {
                SetReward(-0.01f * negativeRewardMultiplier);
                CommitAction(actions);
                CommitReward(GetCumulativeReward());
                MarkAndEndEpisode();
                return;
            }
            if (success)
            {
                float reward = ComputeScoreReward(gameManager.GetCurrentScore());
                SetReward(reward);
            }
            else
            {
                SetReward(-1e-5f * negativeRewardMultiplier);
                RequestDecisionIfNotHeuristic();
            }
            CommitAction(actions);
            CommitReward(GetCumulativeReward());
        }

        #endregion

        #region Game State Callbacks

        public void OnGameStep()
        {
            RequestDecisionIfNotHeuristic();
        }

        public void OnGameOver()
        {
            gameIsOver = true;
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
            gameIsOver = false;
        }

        private static void FillSensorWithGrid(VectorSensor sensor, bool[,] grid)
        {
            // fills grid row-wise, starting with the upper most row
            int width = grid.GetLength(0);
            int height = grid.GetLength(1);
            for (int y = height - 1; y >= 0; y--)
            {
                for (int x = 0; x < width; x++)
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
            return normalizedScoreDelta * positiveRewardMultiplier;
        }
        
        /// <summary>
        /// Checks if the given action is a no-op (contains a negative value).
        /// </summary>
        /// <param name="blockIndex">The index of the block to place.</param>
        /// <param name="x">The x-coordinate of the position to place the block at.</param>
        /// <param name="y">The y-coordinate of the position to place the block at.</param>
        /// <returns>True if the action is a no-op, false otherwise.</returns>
        private static bool IsNopAction(int blockIndex, int x, int y)
        {
            return blockIndex < 0 || x < 0 || y < 0;
        }

        private bool IsHeuristicAgent()
        {
            return behaviorParameters.BehaviorType == BehaviorType.HeuristicOnly;
        }

        private void RequestDecisionIfNotHeuristic()
        {
            if (!isHeuristicAgent)
            {
                RequestDecision();
            }
        }

        private struct UserInput
        {
            public bool hasUserInput;
            public int blockIndex;
            public Vector2Int position;
        }

        #endregion
    }
}
