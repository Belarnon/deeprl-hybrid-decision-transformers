using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Events;
using GameLogic;

namespace PKB.UI
{

    /// <summary>
    /// A <see cref="GridCell"/> is a single cell in the <see cref="GameGrid"/>.
    /// </summary>
    public class GridCell : MonoBehaviour, IDropHandler
    {

        #region Inspector Interface

        /// <summary>
        /// The <see cref="UnityEvent"/> invoked when a block is dropped on this <see cref="GridCell"/>.
        /// </summary>
        /// <remarks>
        /// The first parameter is the index of the Block to select and the second parameter is the coordinates of where it was dropped.
        /// </remarks>
        public UnityEvent<int, Vector2Int> OnBlockDropped;

        #endregion

        #region Internal State

        /// <summary>
        /// The coordinates of this <see cref="GridCell"/> in the <see cref="GameGrid"/>.
        /// </summary>
        private Vector2Int coordinates;

        #endregion

        #region Public Interface

        public void SetCoordinates(Vector2Int coordinates)
        {
            this.coordinates = coordinates;
        }

        #endregion

        #region Unity Event Handlers

        public void OnDrop(PointerEventData eventData)
        {
            GameObject droppedObject = eventData.pointerDrag;
            if (droppedObject == null)
            {
                return;
            }
            if (droppedObject.TryGetComponent(out TileCell tileCell))
            {
                OnBlockDropped?.Invoke(tileCell.GetBlockIndex(), coordinates - tileCell.GetCenterOffset());
            }
        }

        #endregion
    }
}
