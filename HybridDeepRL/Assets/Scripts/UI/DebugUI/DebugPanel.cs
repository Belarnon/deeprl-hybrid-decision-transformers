using UnityEngine;
using UnityEngine.Events;

namespace PKB.UI
{

    /// <summary>
    /// A <see cref="DebugPanel"/> is a panel that can be used to control the game in debug mode.
    /// </summary>
    public class DebugPanel : MonoBehaviour
    {
        #region Inspector Interface

        /// <summary>
        /// This event is invoked when a tile is placed with the debug panel.
        /// </summary>
        [Tooltip("This event is invoked when a tile is placed with the debug panel.")]
        public UnityEvent<int, Vector2Int> onSetTile;

        #endregion

        #region Internal State

        /// <summary>
        /// The currently selected tile to place.
        /// </summary>
        private int blockIndex = 0;

        /// <summary>
        /// The coordinates of where the center block of the current tile will be placed in the grid.
        /// </summary>
        /// <remarks>
        /// X is the horizontal column index from left to right and Y is the vertical row index from bottom to top (both starting at 0).
        /// </remarks>
        /// The coordinate system looks like this:
        /// y
        /// ^
        /// |
        /// |
        /// +------> x
        private Vector2Int tileCoordinates = Vector2Int.zero;

        #endregion

        #region Public Interface

        /// <summary>
        /// Sets the block index to the given value.
        /// </summary>
        /// <param name="value">The new block index.</param>
        public void SetBlockIndex(float value)
        {
            blockIndex = Mathf.RoundToInt(value);
        }

        /// <summary>
        /// Sets the tile row to the given value.
        /// </summary>
        /// <param name="value">The new tile row.</param>
        public void SetTileRow(float value)
        {
            tileCoordinates.y = Mathf.RoundToInt(value);
        }

        /// <summary>
        /// Sets the tile column to the given value.
        /// </summary>
        /// <param name="value">The new tile column.</param>
        public void SetTileColumn(float value)
        {
            tileCoordinates.x = Mathf.RoundToInt(value);
        }

        /// <summary>
        /// Actually places the tile at the given coordinates.
        /// </summary>
        public void PlaceTile()
        {
            onSetTile.Invoke(blockIndex, tileCoordinates);
        }

        #endregion
    }
}
