using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace PKB.Recording
{

    /// <summary>
    /// The <see cref="DemonstrationRecorder"/> can be used to record multiple demonstrated trajectories of an agent.
    /// Specifically, it writes a json file containing the agent's observations, actions, and rewards.
    /// </summary>
    [AddComponentMenu("PKB/Recording/Demonstration Recorder")]
    [RequireComponent(typeof(JSONWriter))]
    public class DemonstrationRecorder : MonoBehaviour
    {
        #region Inspector Interface

        /// <summary>
        /// Whether or not to record.
        /// </summary>
        [Tooltip("Whether or not to record.")]
        [SerializeField]
        private bool record = false;

        /// <summary>
        /// The agent to record.
        /// </summary>
        [Tooltip("The agent to record.")]
        [SerializeField]
        private RecordableAgent agentToRecord;

        #endregion

        #region Internal State

        private JSONWriter jsonWriter;

        private DemonstrationRecord recordToWrite = new();

        private Trajectory currentTrajectory;

        private Transition currentTransition;
        private TransitionFillState currentTransitionFillState = new();

        private bool trajectoryShouldEnd = false;

        #endregion

        #region Unity Lifecycle

        private void Start()
        {
            CheckDependencies();
            jsonWriter = GetComponent<JSONWriter>();
            if (record)
            {
                agentToRecord.OnTrajectoryStarted.AddListener(OnTrajectoryStarted);
                agentToRecord.OnObservationCollected.AddListener(OnObservationCollected);
                agentToRecord.OnActionSelected.AddListener(OnActionSelected);
                agentToRecord.OnRewardReceived.AddListener(OnRewardReceived);
                agentToRecord.OnTrajectoryEnded.AddListener(OnTrajectoryEnded);
            }
        }

        private void OnDestroy()
        {
            if (record)
            {
                agentToRecord.OnTrajectoryStarted.RemoveListener(OnTrajectoryStarted);
                agentToRecord.OnObservationCollected.RemoveListener(OnObservationCollected);
                agentToRecord.OnActionSelected.RemoveListener(OnActionSelected);
                agentToRecord.OnRewardReceived.RemoveListener(OnRewardReceived);
                agentToRecord.OnTrajectoryEnded.RemoveListener(OnTrajectoryEnded);
                AddUnfinishedTrajectory();
                SaveRecord();
            }
        }

        private void AddUnfinishedTrajectory()
        {
            if (currentTrajectory != null)
            {
                recordToWrite.trajectories.Add(currentTrajectory);
            }
        }

        #endregion

        #region Recording Callbacks

        private void OnTrajectoryStarted()
        {
            currentTrajectory = new();
            currentTransition = null;
            currentTransitionFillState.Reset();
        }

        private void OnObservationCollected(VectorSensor sensor)
        {
            EnsureTranitionExists();
            currentTransition.observation = SensorUtils.GetObservationVector(sensor);
            currentTransitionFillState.SetObservationFilled();
            if (LazyEndTrajectory())
            {
                return;
            }
            AddTransitionIfFilled();
        }

        private void OnActionSelected(ActionBuffers actionBuffers)
        {
            EnsureTranitionExists();
            currentTransition.action = CreateActionFromActionBuffer(actionBuffers);
            currentTransitionFillState.SetActionFilled();
            AddTransitionIfFilled();
        }

        private void OnRewardReceived(float reward)
        {
            EnsureTranitionExists();
            currentTransition.reward = reward;
            currentTransitionFillState.SetRewardFilled();
            AddTransitionIfFilled();
        }

        private void OnTrajectoryEnded()
        {
            trajectoryShouldEnd = true;
        }

        #endregion

        #region Helpers

        private void CheckDependencies()
        {
            Debug.Assert(agentToRecord != null, "Agent to record is null.");
        }

        private void EnsureTranitionExists()
        {
            currentTransition ??= new();
        }

        private void AddTransitionIfFilled()
        {
            if (currentTransitionFillState.IsFilled())
            {
                currentTrajectory.transitions.Add(currentTransition);
                currentTransition = null;
                currentTransitionFillState.Reset();
            }
        }

        /// <summary>
        /// Ends the current trajectory if it should end.
        /// </summary>
        /// <remarks>
        /// This is needed as ML-Agents actually sends one last observation, action, and reward after the episode has ended.
        /// </remarks>
        private bool LazyEndTrajectory()
        {
            if (trajectoryShouldEnd && currentTransitionFillState.ObservationFilled)
            {
                trajectoryShouldEnd = false;
                recordToWrite.trajectories.Add(currentTrajectory);
                return true;
            }
            return false;
        }

        private static Action CreateActionFromActionBuffer(ActionBuffers actionBuffers)
        {
            int numDiscreteActions = actionBuffers.DiscreteActions.Length;
            int numContinuousActions = actionBuffers.ContinuousActions.Length;
            Action action = new()
            {
                discreteActions = new int[numDiscreteActions],
                continuousActions = new float[numContinuousActions]
            };
            for (int i = 0; i < numDiscreteActions; i++)
            {
                action.discreteActions[i] = actionBuffers.DiscreteActions[i];
            }
            for (int i = 0; i < numContinuousActions; i++)
            {
                action.continuousActions[i] = actionBuffers.ContinuousActions[i];
            }
            return action;
        }

        private class TransitionFillState
        {
            public bool ObservationFilled = false;
            public bool ActionFilled = false;
            public bool rewardFilled = false;

            public bool IsFilled()
            {
                return ObservationFilled && ActionFilled && rewardFilled;
            }

            public void Reset()
            {
                ObservationFilled = false;
                ActionFilled = false;
                rewardFilled = false;
            }

            public void SetObservationFilled()
            {
                if (ObservationFilled)
                {
                    Debug.LogError("Observation already filled.");
                }
                ObservationFilled = true;
            }

            public void SetActionFilled()
            {
                if (ActionFilled)
                {
                    Debug.LogError("Action already filled.");
                }
                ActionFilled = true;
            }

            public void SetRewardFilled()
            {
                if (rewardFilled)
                {
                    Debug.LogError("Reward already filled.");
                }
                rewardFilled = true;
            }
        }

        #endregion

        #region Helpers

        private void SaveRecord()
        {
            jsonWriter.WriteToFile(recordToWrite);
        }

        #endregion
    }
}
