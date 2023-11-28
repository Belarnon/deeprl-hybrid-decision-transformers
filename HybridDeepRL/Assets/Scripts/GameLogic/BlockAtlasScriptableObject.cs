using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GameLogic
{
	[CreateAssetMenu(fileName = "BlockAtlas", menuName = "Gamelogic/BlockAtlas", order = 2)]
	public class BlockAtlasScriptableObject : ScriptableObject
    {
        public List<BlockScriptableObject> blocks;
    }
}
