using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace PKB.Recording
{
    public abstract class SensorUtils
    {
        public static float[] GetObservationVector(VectorSensor sensor)
        {
            FieldInfo field = sensor.GetType().GetField("m_Observations", BindingFlags.NonPublic | BindingFlags.Instance);
            IList observations = (IList)field.GetValue(sensor);
            float[] observationVector = new float[observations.Count];
            for (int i = 0; i < observations.Count; i++)
            {
                observationVector[i] = (float)observations[i];
            }
            return observationVector;
        }
    }
}
