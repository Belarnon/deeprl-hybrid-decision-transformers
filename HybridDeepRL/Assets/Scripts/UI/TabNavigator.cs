using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace PKB.UI
{
    public class TabNavigator : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
    {
        #region Inspector Interface

        /// <summary>
        /// The list of selectable elements to navigate between using the tab key.
        /// </summary>
        /// <remarks>
        /// The order of the elements in the list determines the order of navigation.
        /// </remarks>
        [Tooltip("The list of selectable elements to navigate between using the tab key.")]
        [SerializeField]
        private List<Selectable> selectabels = new();

        #endregion

        #region Internal State

        private int currentSelectionIndex = 0;

        private bool navigationEnabled = false;

        public void OnPointerEnter(PointerEventData eventData)
        {
            navigationEnabled = true;
        }

        public void OnPointerExit(PointerEventData eventData)
        {
            navigationEnabled = false;
        }

        #endregion

        #region Unity Lifecycle

        private void Start()
        {
            if (selectabels.Count > 0)
            {
                selectabels[0].Select();
            }
            else
            {
                Debug.LogWarning("No selectable elements found in TabNavigator.");
            }
        }

        private void Update()
        {
            if (navigationEnabled)
            {
                if (Input.GetKeyDown(KeyCode.Tab))
                {
                    if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
                    {
                        currentSelectionIndex--;
                        if (currentSelectionIndex < 0)
                        {
                            currentSelectionIndex = selectabels.Count - 1;
                        }
                    }
                    else
                    {
                        currentSelectionIndex++;
                        if (currentSelectionIndex >= selectabels.Count)
                        {
                            currentSelectionIndex = 0;
                        }
                    }
                    selectabels[currentSelectionIndex].Select();
                }
            }
        }

        #endregion
    }
}
