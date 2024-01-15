using UnityEditor;
using PKB.Sensors;
using Unity.MLAgents.Editor;
using Unity.MLAgents.Sensors;

namespace PKB.Editors
{

    [CustomEditor(typeof(BoardSensorComponent))]
    public class BoardSensorComponentEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            SerializedObject serializedObject = base.serializedObject;
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_RenderTexture"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_SensorName"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_Grayscale"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_ObservationStacks"), true);
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_Compression"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("gameManager"), true);
            bool flag = EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
            if (flag)
            {
                UpdateSensor();
            }
        }

        private void UpdateSensor()
        {
            if (base.serializedObject.targetObject is RenderTextureSensorComponent renderTextureSensorComponent)
            {
                // Due to the interal protection level of the UpdateSensor method, we have to use reflection to call it.
                // This is not ideal, but it's the only way to update the sensor.
                var method = typeof(RenderTextureSensorComponent).GetMethod("UpdateSensor", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                method.Invoke(renderTextureSensorComponent, null);
            }
        }
    }
}
